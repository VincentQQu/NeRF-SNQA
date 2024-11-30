nerfstudio_scenes = ["aspen", "bww_entrance", "campanile", "desolation", "dozer", "Egypt", "floating-tree", "Giannini-Hall", "kitchen", "library", "person", "plane", "poster", "redwoods2", "sculpture", "stump", "vegetation"]


nerfstudio_methods = ["splatfacto", "nerfacto", "tensorf", "instant-ngp", "kplanes-20", "kplanes-100", "kplanes-1000", "nerfacto-huge"]



from copy import deepcopy


nerf_methods = {
    "directvoxgo": "DVGO",
    "gnt_crossscene": "GNT-C",
    "gnt_singlescene": "GNT-S",
    "ibrnet_finetune": "IBRNet-S",
    "ibrnet_pretrain": "IBRNet-C",
    "light_field": "LFNR",
    "mipnerf": "MipNeRF",
    "nerf": "NeRF",
    "nex": "NeX",
    "plenoxel": "Plenoxel",
    "reference": "Reference"
}


dataset_names = {"llff": "LLFF", "fieldwork": "Fieldwork", "lab": "Lab"}


dataset_scenes = {
    "llff": {
        "fern": "fern",
        "flower": "flower",
        "fortress": "fortress",
        "horns": "horns",
        "leaves": "leaves",
        "orchids": "orchids",
        "room": "room",
        "trex": "trex",
    },
    "fieldwork": {
        "Bears": "bears",
        "Dinosaur": "dinosaur",
        "Elephant": "elephant",
        "Giraffe": "giraffe",
        "Leopards": "geopards",
        "Naiad_statue": "statue-2",
        "Puccini_statue": "statue-1",
        "Vespa": "vespa",
        "Whale": "whale",
    },
    "lab": {
        "CD_occlusion_extr": "extr-woods",
        "CD_occlusion_intr": "intr-woods",
        # "Ceramic": "ceramic",
        "Glass": "glass",
        "Glossy_animals_extr": "extr-farm",
        "Glossy_animals_intr": "intr-farm",
        "Metal": "metal",
        "Toys": "car-fig",
    },
}

flat_scenes = [f"{ds}+{scene}" for ds in dataset_scenes for scene in dataset_scenes[ds]]


dataset_scenes_to_label = deepcopy(dataset_scenes)
dataset_scenes_to_label["fieldwork"]["Naiad_statue"] = "statue-3"
dataset_scenes_to_label["fieldwork"]["Puccini_statue"] = "statue-2"
dataset_scenes_to_label["fieldwork"]["Vespa"] = "vespa-2"




training_mode = "mixed" # mixed





tr_scenes = {
    "llff": {
        "flower": "flower",
        "fortress": "fortress",
        "leaves": "leaves",
        "horns": "horns",
    },
    "fieldwork": {
        "Dinosaur": "dinosaur",
        "Giraffe": "giraffe",
        "Leopards": "geopards",
        "Naiad_statue": "statue-2",
        "Whale": "whale",
    },
    "lab": {
        "CD_occlusion_intr": "intr-woods",
        "Metal": "metal",
        "Glossy_animals_extr": "extr-farm",
    },
}


tt_scenes = {
    "llff": {
        "fern": "fern",
        "trex": "trex",
        "orchids": "orchids",
        "room": "room",
    },
    "fieldwork": {
        "Bears": "bears",
        "Elephant": "elephant",
        "Puccini_statue": "statue-1",
        "Vespa": "vespa",
    },
    "lab": {
        "CD_occlusion_extr": "extr-woods",
        "Glossy_animals_intr": "intr-farm",
        "Glass": "glass",
        "Toys": "car-fig",
    },
}





eva_metrics = {
    "SRCC": float("-inf"),
    "PLCC": float("-inf"),
}


tr_metrics = {
    "mse": float("inf"), "mae": float("inf"), "loss": float("inf"),
    "SRCC": float("-inf"),
    "PLCC": float("-inf"),
}



def readable_scene(scene):
    scene = scene.capitalize()

    if scene == "Naiad_statue":
        scene = "Naiad-Sta."
    elif scene == "Cd_occlusion_extr":
        scene = "CD-Occ."
    elif scene == "Glossy_animals_intr":
        scene = "Animals"
    
    return scene



def convert_to_readable(nvs_method):
    readable_key = nvs_method.capitalize()

    if readable_key == "Puccini_statue":
        readable_key = "Puccini-Sta."
    elif readable_key == "Cd-occ.":
        readable_key = "Extr-CD-Occ."
    elif readable_key == "Animals":
        readable_key = "Intr-Animals"
    elif readable_key == "Glossy_animals_extr":
        readable_key = "Extr-Animals"
    elif readable_key == "Cd_occlusion_intr":
        readable_key = "Intr-CD-Occ."
    
    return readable_key





dataset_info = {


    "llff": {
        "dataset_folder": "./dataset/nerf_llff_data",
        "image_src": "images_4",
        "tr_keyword": None,
        "labels_path": "./dataset/labels/jod_llff3.csv",
    },
    "fieldwork": {
        "dataset_folder": "./dataset/Fieldwork",
        "image_src": "images",
        "tr_keyword": "_train",
        "labels_path": "./dataset/labels/jod_outdoor3.csv",

    },
    "lab": {
        "dataset_folder": "./dataset/Lab_downscaled4",
        "image_src": "images_4",
        "tr_keyword": "out_", # out_
        "labels_path": "./dataset/labels/jod_lab3.csv",

    },

}




