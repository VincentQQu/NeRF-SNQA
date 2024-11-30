import torch
# import torch.nn as nn
import torch.optim as optim
import random
import glob
import os
# from PIL import Image
from utils import train_utils
import consts
import cv2
import json

from models import sequence_models, contrast_learning, spatio_temporal_model 

from utils.xtimer import Timer, wait_hrs, create_folder
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pending_hr = 0

version = 'v0.0.2'
is_conv1_2d = False

batch_size, n_epochs = 16, 100
n_scene_rep_per_epo, save_per_epo  = 2, 5
min_replace_ratio, add_noise, noise_rnd_range = 0.5, 1, 11
img_mode = "RGB" # "RGB", "LUM"


loss_ws = [1, 0.5, 0.2, 0, 1]
ssim_range = (-4, 1)
psnr_range = (10, 40)

lr=0.001 
crop_range, segment_length_range = [150, 600], [40, 60]


if is_conv1_2d:
    seq_model_config = {
        "img_mode": img_mode, 
        "chs": [3, 12, 24],
        "angular_feat_size": 96,
        "version": version
    }

else:
    seq_model_config = {
        "model_type": "TransEncoder", # LSTM, TransEncoder
        "img_mode": img_mode, 
        "enc_type": "normal", # normal, efficient
        "view_enc_size": 192,
        "with_low_level": [24, 48, 96],
        "view_pretrained": False,
        "angular_feat_size": 384,
        "fusion": "last", # attention, last, mean, max
        "num_layers": 2,
        "nhead": 6,
        "version": version
    }

projector_config = {
"input_dim": 384,
"mlp_dim": 384,
"output_dim": 384,
"version": version
}


max_cache = 200*200*60*16 # modify according to your gpu memory, current max_cache works on 24GB GPU


max_memory = max_cache / batch_size # batch_size=17

loss_keys = ["diff_with_ssim", "diff_with_psnr", "loss_replace", "loss_strred", "loss_vdp", "loss"]

model_save_folder = create_folder(f"./checkpoints/{version}/models")






def get_paired_batch(remain_scenes, methods, batch_size, segment_length_range, device, verbose=True):
    batch1, batch2 = [], []


    selected_scenes = []
    min_seq_len = 1000000
    min_height, min_width = 1000000, 1000000
    for pair_i in range(batch_size):
        scene = random.choice(remain_scenes)
        selected_scenes.append(scene)
        remain_scenes.remove(scene)


        load_folder =  f"./renders/{scene}__{methods[0]}/"
        example_image = cv2.imread(f"{load_folder}00000.png")
        

        height, width = example_image.shape[:2]
        
        img_count = train_utils.count_image_files(load_folder)

        if img_count < min_seq_len: min_seq_len = img_count
        
        if width < min_width: min_width = width
        if height < min_height: min_height = height

    
    min_width = int(min_width - 15)
    min_height = int(min_height - 5)
    

    crop_width = random.randint(*crop_range)
    crop_height = random.randint(*crop_range)
    if crop_width > min_width: crop_width = min_width
    if crop_height > min_height: crop_height = min_height
    crop_size = (crop_height, crop_width)


    segment_length = max_memory // (crop_width*crop_height)
    segment_length = int(segment_length)

    if segment_length > min_seq_len:
        segment_length = min_seq_len

    rotate = False # random.choice([True, False])


    replace_ratios = [min_replace_ratio + random.random()*(1-min_replace_ratio) for _ in range(batch_size)]

    

    
    if verbose: print(f"[{crop_width}x{crop_height}x{segment_length}]", end=': ')
    
    for pair_i, scene in enumerate(selected_scenes):


        method1, method2 = random.sample(methods, 2)




        load_folder =  f"./renders/{scene}__{method1}/"
        example_image = cv2.imread(f"{load_folder}00000.png")

        if verbose: print(f"[{scene}]~[x{replace_ratios[pair_i]:.2f}]({method1}<->{method2})", end='; ')

        transform_params = train_utils.generate_random_params(example_image.shape[:2], crop_size)

        if random.random() < add_noise:
            noise_keys = [random.randint(0, noise_rnd_range), random.randint(0, noise_rnd_range)]
        else:
            noise_keys = [11, 11]

        # print(noise_keys)

        transform_params = [*transform_params, rotate]


        segment1, segment2 = train_utils.load_sequence_segment(load_folder, method1, method2, segment_length, transform_params, noise_keys, img_mode, replace_ratios[pair_i], device)


        # train_utils.save_image_sequences(segment1, segment2, "./check_seg")

        batch1.append(torch.stack(segment1))
        batch2.append(torch.stack(segment2))

    
    if verbose: print()

    batch1 = torch.stack(batch1)
    batch2 = torch.stack(batch2)


    return batch1, batch2, replace_ratios



def train(contrast_learner, optimizer, scenes, methods, batch_size, segment_length_range):

    remain_scenes = (n_scene_rep_per_epo*scenes).copy()
    random.shuffle(remain_scenes)
    n_batches = len(remain_scenes) // batch_size

    avg_loss = {lk: 0 for lk in loss_keys}

    for b_i in range(1, n_batches + 1):
        # Sample batch

        batch1, batch2, replace_ratios = get_paired_batch(remain_scenes, methods, batch_size, segment_length_range, device)

        batch1, batch2  = batch1.to(device), batch2.to(device)

        # print(batch1)

        replace_ratios = torch.tensor(replace_ratios, device=device)
        

        # Forward pass
        optimizer.zero_grad()

        loss, loss_parts = contrast_learner.loss(batch1, batch2, replace_ratios)

        # print(loss)

        loss.backward()
        optimizer.step()


        for lk in loss_keys:
            avg_loss[lk] += loss_parts[lk]


        print(f"b_{b_i:<4d}", '-'*3, end=' ')
        print(train_utils.mini_fmt_loss_equ(loss_parts) + f" [{b_i:>6d}/{n_batches:>6d}]")
        # tim.lap()

        torch.cuda.empty_cache()
        gc.collect()


    for lk in loss_keys:
        avg_loss[lk] = avg_loss[lk] / n_batches
    

    return avg_loss






def main_func():


    ns_scenes = consts.nerfstudio_scenes
    ns_methods = consts.nerfstudio_methods
    

    epo_offset = 0
    

    # uncomment the following to resume an interrupted training
    # modify epo_offet to your last epo

    # epo_offset = 55
    # global n_epochs
    # n_epochs -= epo_offset

    if epo_offset != 0:
        encoder, projector = train_utils.load_model(epo_offset, model_save_folder, device, only_encoder=False, is_conv1_2d=is_conv1_2d)

        log_path = os.path.join(model_save_folder, f"log_{encoder.model_name}.json") 

        with open(log_path) as infile:
            loss_log = json.load(infile)
            loss_log = train_utils.clear_log_after(loss_log, epo_offset, loss_keys)
    
    else:
        
        

        if is_conv1_2d:
            encoder = spatio_temporal_model.ResNet_R2Plus1D_LN(**seq_model_config).to(device)
        else:
            encoder = sequence_models.EfficientSequenceModel(**seq_model_config).to(device)

        encoder.save_config(model_save_folder)

        projector = contrast_learning.MLPs(projector_config, num_mlp=len(loss_ws)).to(device)
        projector.save_config(model_save_folder)


        log_path = os.path.join(model_save_folder, f"log_{encoder.model_name}.json")
        loss_log = train_utils.init_log(loss_keys)
    

    encoder.train()
    projector.train()


    print(encoder.model_name)
    num_params = contrast_learning.count_parameters(encoder)
    print(f"num of parameters: {num_params:,}")

    print(projector.model_name)
    num_params = contrast_learning.count_parameters(projector)
    print(f"num of parameters: {num_params:,}")

    loss_log["loss_ws"] = loss_ws
    loss_log["min_replace_ratio"] = min_replace_ratio




    contrast_learner = contrast_learning.ContrastiveLearning(encoder, projector, ssim_range, psnr_range, loss_ws).to(device)
    optimizer = optim.Adam(contrast_learner.parameters(), lr=lr, amsgrad=True)


    tt_tr_msg = "to_be_updated"
    show_table_msg = "to_be_updated"

    better_count = -1

    for epo_i in range(1, n_epochs+1):

        real_epo = epo_i+epo_offset
        epo_perc = (epo_i-1+0.0115)/n_epochs
        time_so_far = tim.total_t() /3600
        time_left = time_so_far / epo_perc - time_so_far
        print("-"*30 + f" Epoch {real_epo} / {n_epochs+epo_offset} ~{epo_perc*100:.1f}% ~{time_so_far:.2f}hr<{time_left:.2f}hr "+ "-"*30)

        avg_loss = train(contrast_learner, optimizer, ns_scenes, ns_methods, batch_size, segment_length_range)



        for lk in loss_keys:
            loss_log[lk].append(avg_loss[lk])
        
        with open(log_path, 'w') as outfile:
            json.dump(loss_log, outfile, indent=4)


        print("="*100)
        print(f"epo_{epo_i:<4d}", '-'*3, end=' ')
        print(train_utils.mini_fmt_loss_equ(avg_loss) + f" [{epo_i:>6d}/{n_epochs:>6d}]")

        tim.lap()


        encoder_path = os.path.join(model_save_folder, f"{encoder.model_name}_realtime.pth") 
        torch.save(encoder.state_dict(), encoder_path)

        projector_path = os.path.join(model_save_folder, f"{projector.model_name}_realtime.pth") 
        torch.save(projector.state_dict(), projector_path)

        if epo_i < 5 or epo_i % save_per_epo == 0:
            encoder_path = os.path.join(model_save_folder, f"{encoder.model_name}_{real_epo}.pth") 
            torch.save(encoder.state_dict(), encoder_path)

            projector_path = os.path.join(model_save_folder, f"{projector.model_name}_{real_epo}.pth") 
            torch.save(projector.state_dict(), projector_path)


            print(f"Saved model at epoch {real_epo}.")


            
            time_so_far = tim.total_t() /3600
            current_epo = {"epo_n": real_epo, "tr_time": time_so_far, "tr": {key: avg_loss[key] for key in loss_keys}}

            tt_tr_msg = f"current_epo [{real_epo}]:\ntr~{current_epo['tr']['diff_with_ssim']:>8.4f}~diff_with_ssim|{current_epo['tr']['diff_with_psnr']:>8.4f}~diff_with_psnr|{current_epo['tr']['loss_replace']:>8.4f}~loss_replace|{current_epo['tr']['loss_strred']:>8.4f}~loss_strred|{current_epo['tr']['loss_vdp']:>8.4f}~loss_vdp|{current_epo['tr']['loss']:>8.4f}~loss\n"
            print(tt_tr_msg)


  
            
            tim.lap()
        


        print(encoder.model_name)

        print()

        torch.cuda.empty_cache()
        gc.collect()



            

    print(f"Done! ~{tim.total_t() /3600:.2f}hr")


if __name__ == "__main__":
    
    wait_hrs(pending_hr)

    tim = Timer()

    tim.start()


    error_count = 0
    no_err = False
    while not no_err:
        try:
            main_func()
            no_err = True
        except AssertionError as e:
            print(str(e))
            error_count += 1
            print(f"({error_count}) times assertion error.")
            # exit()

    tim.stop()