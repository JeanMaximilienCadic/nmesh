from .converter import Converter
import os
from gnutools.fs import parent, name
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from nmesh import load_config


def main(folder_xy_ply, folder_xy_xyz, resolution=64, verbose=False):
    file = f"{folder_xy_xyz}/{resolution}/{name(folder_xy_ply)}/y.xyx"
    try:
        assert not os.path.exists(file)
        os.makedirs(parent(file), exist_ok=True)
        compressor = Converter(folder_xy_ply, resolution=resolution)
        compressor.export(root_output=folder_xy_xyz)
        print('++ {output_file}'.format(output_file=file)) if verbose else None
    except AssertionError:
        pass
    except:
        with open("errors.csv", "a") as f:
            f.write(f"{folder_xy_ply}\n")


if __name__ == "__main__":
    args = load_config()
    with ProcessPoolExecutor() as e:
        fs = [e.submit(main,
                       folder_xy_ply=f"{args.root_xy_ply}/{folder_name}",
                       folder_xy_xyz=f"{args.root_xy_xyz}/{folder_name}")
              for folder_name in os.listdir(args.root_xy_ply)]
        for f in tqdm(as_completed(fs), total=len(fs), desc="Processing"):
            assert f._exception is None