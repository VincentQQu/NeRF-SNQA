from utils import train_utils
import numpy as np
import torch
import os
import glob
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate_quality_features(epo_offset, model_save_folder, eval_folder, device):
    encoder = train_utils.load_model(epo_offset, model_save_folder, device, only_encoder=True, is_conv1_2d=False)
        
    encoder.eval()

    feats_save_name = f"output_features.npz"


    feats_save_path = os.path.join(eval_folder, feats_save_name)

    # if os.path.exists(feats_save_path):
    #     return np.load(feats_save_path), feats_save_path

    all_feats = {}

    nss_names = [ name for name in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, name)) ]

    nss_names =  sorted(nss_names)

    for name in nss_names:
        nss_folder = os.path.join(eval_folder, name)

        view_paths = glob.glob(os.path.join(nss_folder, "*.*"))

        view_paths = [vp for vp in view_paths if vp.split(".")[-1] in ["png", "jpg"] ]

        view_paths = sorted(view_paths)

        with torch.no_grad():
            feats = encoder.predict_features_from_view_paths(view_paths, device)

        all_feats[name] = feats.cpu().numpy()
    

    np.savez(feats_save_path, **all_feats)

    print(f"Quality features were saved in {feats_save_path}")


    return all_feats, feats_save_path





def generate_quality_scores(all_feats, model_save_folder):

    print("========= Calculating Quality Scores ==========")
    print("Note 1: JOD, the chosen scoring format, features primarily negative scores, where higher values denote better quality.")
    print("Note 2: According to the dataset's author, JOD scores hold greater relevance within the same scene, rendering cross-scene comparisons of JOD potentially less meaningful.")

    quality_scores = {}

    regr_save_folder = model_save_folder.replace("models", "regr")

    regr_save_path = os.path.join(regr_save_folder, "linear_regr.pkl")
    with open(regr_save_path, 'rb') as f:
        regr = pickle.load(f)
    
    tt_x = [ all_feats[name] for name in all_feats]

    preds = regr.predict(tt_x)

    print("========= Quality Score Results ==========")

    for idx, nss_name in enumerate(all_feats):
        pred = preds[idx]
        print(f"{nss_name}: {pred:.4f}")
    
        quality_scores[nss_name] = pred


    return quality_scores




def main():
    version = "v0.0.1"
    model_save_folder = f"./checkpoints/{version}/models"

    eval_folder = f"./examples/"

    all_feats, feats_save_path = generate_quality_features(epo_offset=100, model_save_folder=model_save_folder, eval_folder=eval_folder, device=device)


    quality_scores = generate_quality_scores(all_feats, model_save_folder)

    # quality_scores stores the quality score in the following format
    # {'Giraffe+NeRF': -2.0440, 'Giraffe+IBRNet-C': -2.1063}

    # print(quality_scores)



if __name__ == "__main__":
    main()