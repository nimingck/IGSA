import importlib

attack_zoo = {
    'dim':('.input_transformation.dim', "DIM"),
    'fgsm': ('.gradient.fgsm', "FGSM"),
    'mifgsm': ('.gradient.mifgsm', "MIFGSM"),
    'raa': ('.gradient.raa', "RAA"),
    'raa_dta': ('.gradient.raa_dta', "RAA"),
    'raa_smifgrm': ('.gradient.raa_smifgrm', "RAA"),
    'raa_input': ('.gradient.raa_input', "RAA"),
    'raa_face': ('.gradient.raa_face', "RAA"),
    'smifgrm': ('.gradient.smifgrm', "SMIFGRM"),


    
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception('Unspported attack algorithm {}'.format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class


__version__ = '1.0.0'
