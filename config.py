from os import path

root_dir = path.dirname(path.abspath(__file__))

# Set the path of observations data
if path.isdir(f"{root_dir}/data/"):
    data_dir = root_dir  # On local computer
else:
    data_dir = "/data/cmathe/observational_data/"  # On CICLAD
