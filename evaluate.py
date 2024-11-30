from utils import train_utils, benchmarking_utils
import numpy as np
import torch
import consts
import os
from sklearn import linear_model
import glob
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def generate_features(encoder, epo_offset, model_save_folder, is_conv1_2d, device, with_projector=False):
    if encoder == None:

        encoder = train_utils.load_model(epo_offset, model_save_folder, device, only_encoder=True, is_conv1_2d=is_conv1_2d)

    encoder.eval()

    feats_save_name = f"feats_{encoder.model_name}_{epo_offset}.npz"


    output_folder = model_save_folder.replace("models", "outputs")

    if not os.path.exists(output_folder): os.makedirs(output_folder)
    feats_save_path = os.path.join(output_folder, feats_save_name)

    if os.path.exists(feats_save_path):
        # print(feats_save_path)
        return np.load(feats_save_path), feats_save_path # , allow_pickle=True


    all_feats = {}

    for ds in consts.dataset_scenes:
        for scene in consts.dataset_scenes[ds]:
            for method in consts.nerf_methods:

                if method == "reference": continue
                seq_folder = os.path.join("benchmark_bank", ds, scene, method, "frames")


                view_paths = glob.glob(os.path.join(seq_folder, "*.*"))

                view_paths = [vp for vp in view_paths if vp.split(".")[-1] in ["png", "jpg"] ]

                view_paths = sorted(view_paths)

                # print(view_paths)

                with torch.no_grad():

                    feats = encoder.predict_features_from_view_paths(view_paths, device)

                all_feats[f"{ds}+{scene}+{method}"] = feats.cpu().numpy()
    

    np.savez(feats_save_path, **all_feats)

    return all_feats, feats_save_path





def regr_core(all_feats, tr_keys, tt_keys, tr_y, regr_save_path=None):


    tr_x = [all_feats[tk] for tk in tr_keys]
    
    tr_x = np.array(tr_x)#.reshape(-1, 1)

    tt_x = [all_feats[tk] for tk in tt_keys]
    # print(tt_x[100].shape)
    tt_x = np.array(tt_x)#.reshape(-1, 1)

    regr = build_regr_model() 


    regr.fit(tr_x, tr_y)

    if regr_save_path != None:
        with open(regr_save_path,'wb') as f:
            pickle.dump(regr,f)

    preds = regr.predict(tt_x)

    preds = preds.tolist()

    return preds


def build_regr_model():
    regr = linear_model.Ridge()
    return regr






def regress_features(encoder, epo_offset, model_save_folder, is_conv1_2d, device):


    all_feats, feats_save_path = generate_features(encoder,epo_offset, model_save_folder, is_conv1_2d, device)

    tr_keys, tt_keys, labels = benchmarking_utils.load_flat_labels_scenewise_split()


    tr_y = np.array([labels[k] for k in tr_keys])
    tt_y = [labels[k] for k in tt_keys]

    regr_save_folder = model_save_folder.replace("models", "regr")
    if not os.path.exists(regr_save_folder): os.makedirs(regr_save_folder)
    regr_save_path = os.path.join(regr_save_folder, "linear_regr.pkl")

    preds = regr_core(all_feats, tr_keys, tt_keys, tr_y, regr_save_path)



    # qa_save_path = feats_save_path.replace("feats_", "qa_").replace(".npz", "")


    output_folder = model_save_folder.replace("models", "outputs")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    names_preds_trues = benchmarking_utils.get_names_preds_trues(tt_keys, preds, tt_y)

    # print(names_preds_trues)

    # benchmarking_utils.save_names_preds_trues(names_preds_trues, qa_save_path)


    results_group_by_dataset, _ = benchmarking_utils.get_grouped_table(names_preds_trues, group_by="dataset", dec_plc=4)

    results_group_by_dataset["Average"] = benchmarking_utils.calculate_grouped_results_avg(results_group_by_dataset)

    # table_pth = qa_save_path + ".txt"
    show_table_msg = '\n'

    mini_table_msg = '\n'

    group_tables = {"dataset": "\n", "scene": "\n"}

    for gt in group_tables:
        grouped_results, group_tables[gt] = benchmarking_utils.get_grouped_table(names_preds_trues, group_by=gt, dec_plc=4)

        show_table_msg += f"Group by {gt}\n"
        show_table_msg += group_tables[gt]
        show_table_msg += '\n'

        if gt == "dataset":
            mini_table_msg += group_tables[gt]

        # with open(table_pth, "w") as table_file:
        #     table_file.write(show_table_msg)

    return show_table_msg, results_group_by_dataset

    


# def get_preds_from_ssl_nerf_nqa(version, epo_offset, tr_keys, tt_keys, labels, datasetwise=False):

#     model_save_folder = f"./checkpoints/{version}/models"

#     all_feats, feats_save_path = generate_features(encoder=None,epo_offset=epo_offset, model_save_folder=model_save_folder, is_conv1_2d=False, device=device)

#     tr_y = np.array([labels[k] for k in tr_keys])
#     # tt_y = [labels[k] for k in tt_keys]

#     preds = regr_core(all_feats, tr_keys, tt_keys, tr_y, datasetwise, model_save_folder)

#     return preds





if __name__ == "__main__":
    version = "v0.0.1"
    model_save_folder = f"./checkpoints/{version}/models"

    show_table_msg, results_group_by_dataset = regress_features(encoder=None, epo_offset=100, model_save_folder=model_save_folder,is_conv1_2d=False, device=device)
    print("========= One Linear Model for Three Datasets =========")
    print(show_table_msg)


