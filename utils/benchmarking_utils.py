import consts
import os, shutil, glob
from utils.xtimer import Timer
import cv2
# import random
import json
import csv
from tabulate import tabulate
import numpy as np
from scipy.stats import pearsonr, spearmanr
import random


# gathering benchmarking bank
# ./benchmarking_bank/llff/scene/method/video_frames_images


# llff: /benchmarking_bank/llff/scene/method/frames or images



def extract_frames(video, save_folder):
    vidcap = cv2.VideoCapture(video)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(save_folder,"frame{:07d}.jpg".format(count)), image)     # save frame as JPEG file
        count += 1
    print(f"{count} images are extacted in {save_folder}.")




def generate_frames_videos(dataset):
    # define dataset folder
    dataset_folder = consts.dataset_info[dataset]["dataset_folder"]

    video_folder = "./dataset/benchmark_videos"


    scene_names = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    scene_names = sorted(scene_names)


    benchmark_folder = f"./benchmark_bank/{dataset}"



    for scene in scene_names:

        print("="*20, f"extracting video frames for [{scene}]", "="*20)

        scene_folder = os.path.join(benchmark_folder, scene)

        if not os.path.exists(scene_folder):
            os.makedirs(scene_folder)
        # else:
        #     shutil.rmtree(frames_folder)
        
        video_scene = consts.dataset_scenes[dataset][scene]

        video_paths = glob.glob(f"{video_folder}/scene={video_scene},method=*.mp4")
        video_paths = sorted(video_paths)

        for video in video_paths:
            method_start_i = video.find("method=") + 7
            method_end_i = video.find(".mp4")
            method = video[method_start_i:method_end_i]

            method_folder = os.path.join(scene_folder, method)

            if not os.path.exists(method_folder):
                os.makedirs(method_folder)

            frame_folder = os.path.join(method_folder, "frames")

            # if not os.path.exists(frame_folder):
            #     os.makedirs(frame_folder)
            
            save_video_path = os.path.join(method_folder, "video.mp4")

            shutil.copy2(video, save_video_path)

            # extract_frames(video, frame_folder)

            frame_read_folder = os.path.join(dataset_folder, scene, "frames", method)

            shutil.copytree(frame_read_folder, frame_folder)

    



# def gather_bank_for_a_dataset(dataset):

def gather_llff_reference(dataset="llff"):
    dataset_folder = consts.dataset_info[dataset]["dataset_folder"]

    scene_names = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    scene_names = sorted(scene_names)
    
    image_src = consts.dataset_info[dataset]["image_src"]

    benchmark_folder = f"./benchmark_bank/{dataset}"


    for scene in scene_names:
        image_folder = os.path.join(dataset_folder, scene, image_src)

        dest_dir = os.path.join(benchmark_folder, scene, "reference", "images")

        # if not os.path.exists(dest_dir):
        #     os.makedirs(dest_dir)

        shutil.copytree(image_folder, dest_dir)

    


def generate_llff_sparse_image(dataset="llff"):
    benchmark_folder = f"./benchmark_bank/{dataset}"

    scene_names = [d for d in os.listdir(benchmark_folder) if os.path.isdir(os.path.join(benchmark_folder, d))]
    scene_names = sorted(scene_names)

    src_folder = f"./data_src/llff_sparse_images"
    methods = os.listdir(src_folder)



    for scene in scene_names:
        scene_folder = os.path.join(benchmark_folder, scene)

        # methods = os.listdir(scene_folder)

        # ref_folder = os.path.join(scene_folder, "reference", "images")

        # ref_images = sorted([f for f in os.listdir(ref_folder) if f.endswith(('.jpg', '.jpeg', '.png','.JPG', '.PNG'))])
        

        for mtd in methods:

            # if mtd == "reference": continue

            method_folder = os.path.join(scene_folder, mtd)

            # frame_folder = os.path.join(method_folder, "frames")

            # frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(('.jpg', '.jpeg', '.png','.JPG', '.PNG'))])

            image_folder = os.path.join(method_folder, "images")

            src_path = os.path.join(src_folder, mtd, scene)

            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            

            shutil.copytree(src_path, image_folder, dirs_exist_ok = True)
            




def gather_bank():
    tim = Timer()
    tim.start()

    for dataset in consts.dataset_names:
        generate_frames_videos(dataset)
        tim.lap()
    

    # gather_llff_reference()
    # tim.lap()

    generate_llff_sparse_image()
    tim.lap()




def flatten_all_labels(all_labels):
    tr_keys, tt_keys = [], []
    flat_labels = {"labels": {}}
    for ds in all_labels:
        for scene in all_labels[ds]:
            for method in all_labels[ds][scene]:
                
                if method == "reference": continue
                key = f"{ds}+{scene}+{method}"
                if scene in consts.tr_scenes[ds]:
                    tr_keys.append(key)
                else:
                    tt_keys.append(key)
                flat_labels["labels"][key] = all_labels[ds][scene][method]
                    
    flat_labels["tr_keys"] = tr_keys
    flat_labels["tt_keys"] = tt_keys
    return flat_labels




def load_flat_labels_scenewise_split(tr_ratio=0.5, random_seed=11):


    flat_labels_path = os.path.join("benchmark_bank", f"flat_labels_offset_by_ref_scenewise_split_{random_seed}.json")


    if os.path.exists(flat_labels_path):
        with open(flat_labels_path, 'r') as openfile:
            # Reading from json file
            flat_labels = json.load(openfile)
            tr_keys = flat_labels["tr_keys"]
            tt_keys = flat_labels["tt_keys"]
            labels = flat_labels["labels"]
    
    else:

        # Set the random seed
        random.seed(random_seed)

        labels_path = os.path.join("benchmark_bank", "all_labels_offset_by_ref.json")

        # Read the nested dictionary from the JSON file
        with open(labels_path, 'r') as file:
            nested_dict = json.load(file)

        labels = {}
        tr_keys = []
        tt_keys = []

        # Flatten the dictionary
        for dataset, scenes in nested_dict.items():
            for scene, methods in scenes.items():
                for method, label in methods.items():
                    if method == "reference": continue
                    key = f"{dataset}+{scene}+{method}"
                    labels[key] = label

        # Split the keys for each scene
        for scene in set(key.split('+')[1] for key in labels.keys()):
            scene_keys = [key for key in labels if f'+{scene}+' in key]
            random.shuffle(scene_keys)
            split_point = int(len(scene_keys) * tr_ratio)
            tr_keys += scene_keys[:split_point]
            tt_keys += scene_keys[split_point:]

        # Save the outputs to JSON files

        flat_labels = {}
        flat_labels["tr_keys"] = tr_keys
        flat_labels["tt_keys"] = tt_keys
        flat_labels["labels"] = labels
        
        with open(flat_labels_path, 'w') as outfile:
            json.dump(flat_labels, outfile, indent=4)

    return tr_keys, tt_keys, labels

   



def load_flat_labels():
    flat_labels_path = os.path.join("benchmark_bank", "flat_labels_offset_by_ref_nerfnqa_split.json")

    

    if os.path.exists(flat_labels_path):
        with open(flat_labels_path, 'r') as openfile:
            # Reading from json file
            flat_labels = json.load(openfile)
            tr_keys = flat_labels["tr_keys"]
            tt_keys = flat_labels["tt_keys"]
            labels = flat_labels["labels"]
    
    else:

        labels_path = os.path.join("benchmark_bank", "all_labels_offset_by_ref.json")

        with open(labels_path, 'r') as openfile:
            # Reading from json file
            all_labels = json.load(openfile)
            flat_labels = flatten_all_labels(all_labels)
            tr_keys = flat_labels["tr_keys"]
            tt_keys = flat_labels["tt_keys"]
            labels = flat_labels["labels"]
        
        with open(flat_labels_path, 'w') as outfile:
            json.dump(flat_labels, outfile, indent=4)
    

    return tr_keys, tt_keys, labels





def save_dict_to_csv(data_dict, csv_file_path):
    """
    Save a dictionary to a CSV file where values are lists representing columns.

    Parameters:
    - data_dict: Dictionary to save
    - csv_file_path: Path to the CSV file
    """
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header (keys of the dictionary)
        writer.writerow(data_dict.keys())
        
        # Transpose the dictionary values to write rows
        for row in zip(*data_dict.values()):
            writer.writerow(row)


def get_names_preds_trues(names, preds, trues):

  if type(preds) != list:
    preds = preds.tolist()
    trues = trues.tolist()


  assert len(preds) == len(trues), "len(preds) != len(trues)"

  assert len(preds) == len(names), "len(preds) != len(names)"


  names_preds_trues = {
    "names": names,
    "preds": preds,
    "trues": trues
  }

  return names_preds_trues



def get_grouped_table(names_preds_trues, group_by="dataset", dec_plc=4):

  grouped_output = {}

  for name, pred, true in zip(names_preds_trues["names"], names_preds_trues["preds"], names_preds_trues["trues"]):

    # print(name.split('+'))
    ds, scene, mhd = name.split('+')

    if group_by == "dataset":
      # k = consts.dataset_names[ds]

      k = f"{consts.dataset_names[ds]}+{consts.readable_scene(scene)}"

    elif group_by == "method":
      k = consts.nerf_methods[mhd]
    elif group_by == "scene":
      k = consts.readable_scene(scene)
      # k = scene.capitalize()




    if k not in grouped_output: grouped_output[k] = {"preds": [], "trues": []}

    grouped_output[k]["preds"].append(pred)
    grouped_output[k]["trues"].append(true)


  grouped_results = {}
  for k in grouped_output:
    preds = np.array(grouped_output[k]["preds"])
    trues = np.array(grouped_output[k]["trues"])

    grouped_results[k] = calculate_eva_results(preds, trues)
  

  if group_by == "dataset":
    unflatten_grouped_results = {}
    for k in grouped_results:
      ds, scene = k.split('+')
      if ds not in unflatten_grouped_results:
          unflatten_grouped_results[ds] = {}

      unflatten_grouped_results[ds][scene] = grouped_results[k]

    new_grouped_results = {}
    for ds in consts.dataset_names.values():
      new_grouped_results[ds] = calculate_grouped_results_avg(unflatten_grouped_results[ds])
    grouped_results = new_grouped_results

  return grouped_results, show_metrics(grouped_results, dec_plc)



def process_metrics():
    
    output_file_path = os.path.join("expr_results", "best_table_results_mixed.json")
    
    json_file_path = os.path.join("expr_results", "table_results_mixed.json")
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize the structure for best and second-best values
    results = {"best": {}, "second": {}}

    # Define a helper function to update the best and second-best values
    def update_results(group_by, key, metric, value, is_lower_better):
        if group_by not in results["best"]:
            results["best"][group_by] = {}
            results["second"][group_by] = {}
        if key not in results["best"][group_by]:
            results["best"][group_by][key] = {}
            results["second"][group_by][key] = {}
        
        best_value = results["best"][group_by][key].get(metric, float('inf') if is_lower_better else float('-inf'))
        second_best_value = results["second"][group_by][key].get(metric, float('inf') if is_lower_better else float('-inf'))

        # Update logic based on whether lower or higher values are better
        if (is_lower_better and value < best_value) or (not is_lower_better and value > best_value):
            # Update second-best before best if the new value becomes the best
            results["second"][group_by][key][metric] = best_value
            results["best"][group_by][key][metric] = value
        elif (is_lower_better and best_value < value < second_best_value) or (not is_lower_better and best_value > value > second_best_value):
            results["second"][group_by][key][metric] = value

    # Iterate over the nested structure and update best and second-best values
    for method, group_by_data in data.items():
        
        if "SSL-NeRF" in method or "LFACon" in method: continue
        for group_by, key_data in group_by_data.items():
            for key, metrics in key_data.items():
                for metric, value in metrics.items():
                    is_lower_better = metric in ["RMSE", "OR"]
                    update_results(group_by, key, metric, value, is_lower_better)

    # Save the result to a JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    return results


def calculate_eva_results(preds, trues):
    
    srcc = spearmanr(preds, trues).statistic
    plcc = pearsonr(preds, trues).statistic
    results = {
      "SRCC": srcc,
      "PLCC": plcc,
    }

    return results



def show_metrics(metric_dic, dec_plc):
  rshp_dic = reshape_dic(metric_dic)

  table_val = []

  
  
  for mn in rshp_dic:
    dic_ks = sorted(rshp_dic[mn].keys())
    row_val = [mn]
    for sn in dic_ks: # sorted(
      cell_val = f"{rshp_dic[mn][sn]:.{dec_plc}f}"
      row_val.append(cell_val)
    
    table_val.append(row_val)
  
  table = tabulate(table_val, headers=dic_ks, tablefmt="pretty")  # sorted(tablefmt="pretty")

  return table




def reshape_dic(dic):
  new_dic = {}
  # sorted(
  k1s = list(dic.keys())
  k1s = k1s[1:] + [k1s[0]]
  k2s = dic[k1s[0]].keys()
  for k2 in k2s:
    new_dic[k2] = {}
    for k1 in k1s:
      new_dic[k2][k1] = dic[k1][k2]
  
  return new_dic



def outlier_ratio(data):
    a = np.array(data)
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    iqr = q3 - q1
    lft = q1 - 1.5*iqr
    rgt = q3 + 1.5*iqr

    out_n = 0
    for n in data:
        if n < lft or n > rgt:
            out_n += 1
    return out_n/len(data)




def calculate_grouped_results_avg(grouped_results):



  results = {key: 0 for key in consts.eva_metrics}


  for k in grouped_results:
    for mk in grouped_results[k]:
      results[mk] += grouped_results[k][mk] 
    
  
  for mk in grouped_results[k]:
    results[mk] /= len(grouped_results.keys())
  

  return results




def save_names_preds_trues(names_preds_trues, output_json_path):
  json_path = output_json_path + ".json"

  with open(json_path, 'w') as outfile:
    json.dump(names_preds_trues, outfile, indent=4)
  
  diff = [pred-true for pred, true in zip(names_preds_trues["preds"], names_preds_trues["trues"])]

  names_preds_trues["diff"] = diff

  csv_file_path = output_json_path + ".csv"

  save_dict_to_csv(names_preds_trues, csv_file_path)


  return