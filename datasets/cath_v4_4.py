import functools
import json
import os
import random
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional

import omegaconf
import torch
import wget
from graphein.protein.tensor import Protein
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger
from tqdm import tqdm

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

def _process_single_pdb(args):
    """
    Worker for converting one PDB to a .pt. Returns (name, status, err_msg).
    """
    pdb_path, processed_dir, overwrite = args
    pdb_path = Path(pdb_path)
    processed_dir = Path(processed_dir)
    out_path = processed_dir / f"{pdb_path.stem}.pt"

    if out_path.exists() and not overwrite:
        return pdb_path.name, "skipped", None

    try:
        prot = Protein().from_pdb_file(str(pdb_path))
        torch.save(prot, str(out_path))
        return pdb_path.name, "converted", None
    except Exception as e:
        return pdb_path.name, "failed", str(e)


class CATHDataModule(ProteinDataModule):
    """Data module for CATH dataset.

    :param path: Path to store data.
    :type path: str
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :param format: Format to load PDB files in.
    :type format: Literal["mmtf", "pdb"]
    :param pdb_dir: Path to directory containing PDB files.
    :type pdb_dir: str
    :param pin_memory: Whether to pin memory for dataloaders.
    :type pin_memory: bool
    :param in_memory: Whether to load the entire dataset into memory.
    :type in_memory: bool
    :param num_workers: Number of workers for dataloaders.
    :type num_workers: int
    :param dataset_fraction: Fraction of dataset to use.
    :type dataset_fraction: float
    :param transforms: List of transforms to apply to dataset.
    :type transforms: Optional[List[Callable]]
    :param overwrite: Whether to overwrite existing data.
        Defaults to ``False``.
    :type overwrite: bool
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        format: Literal["pdb"] = "pdb",
        pdb_dir: Optional[str] = None,
        pin_memory: bool = True,
        in_memory: bool = False,
        num_workers: int = 16,
        dataset_fraction: float = 1.0,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir = Path(path)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.format = format
        self.pdb_dir = pdb_dir
        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir, exist_ok=True)
            logger.warning(f"PDB directory {self.pdb_dir} did not exist and has been created. ")

        self.dataset_fraction = dataset_fraction
        self.excluded_chains: List[str] = self.exclude_pdbs()

        self.prepare_data_per_node = False

    def _check_all_processed_files_exist(self) -> bool:
        """Check if all required .pt files exist in processed directory."""
        if not (self.data_dir / "chain_set_splits.json").exists():
            logger.warning("chain_set_splits.json not found, prepare download...")
            return False

        with open(self.data_dir / "chain_set_splits.json", "r") as f:
            splits = json.load(f)

        all_chains = splits["train"] + splits["validation"] + splits["test"]
        for chain in all_chains:
            if not (self.processed_dir / f"{chain.split('.')[0]}.pt").exists():
                logger.warning(f"Processed file for {chain} not found, prepare download...")
                return False
        return True

    def download(self):
        if self._check_all_processed_files_exist():
            logger.info("Found all required processed files, skipping download and preparation")
            return

        self._download_and_prepare_cath(threshold=40)
        self._convert_all_pdbs_to_tensors(overwrite=False, n_procs=self.num_workers)
        self._train_test_split(threshold=40, seed=43)

    def parse_labels(self):
        """Not implemented for CATH dataset"""
        pass

    def exclude_pdbs(self):
        """Not implemented for CATH dataset"""
        return []



    def _download_and_prepare_cath(self, threshold: int = 40):
        """
        Download CATH 4.4 non-redundant S{threshold} files, extract PDBs,
        flatten the extracted 'dompdb' folder into its parent, delete it,
        normalize extensions, and build chain_set_splits.json.
        """
        base_url = (
            "ftp://orengoftp.biochem.ucl.ac.uk/"
            "cath/releases/latest-release/non-redundant-data-sets"
        )
        names = [
            f"cath-dataset-nonredundant-S{threshold}.list",
            f"cath-dataset-nonredundant-S{threshold}.pdb.tgz",
        ]

        # 1) Download the .list and .pdb.tgz
        for fname in names:
            dest = self.data_dir / fname
            if not dest.exists():
                logger.info(f"Downloading {fname} …")
                wget.download(f"{base_url}/{fname}", str(dest))
                logger.info(f" → saved to {dest}")
            else:
                logger.info(f"Found existing file, skipping: {fname}")

        # 2) Extract the .pdb.tgz into data_dir
        tgz = self.data_dir / names[-1]
        extract_root = self.raw_dir
        logger.info(f"Extracting all PDBs from {tgz} → {extract_root}")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(path=extract_root)
        logger.info("Extraction complete.")

        # 2b) Flatten 'dompdb': move all its contents up one level, then remove it
        dompdb_dir = extract_root / "dompdb"
        if dompdb_dir.exists() and dompdb_dir.is_dir():
            for item in dompdb_dir.iterdir():
                target = extract_root / item.name
                item.rename(target)
            dompdb_dir.rmdir()
            logger.info(f"Moved contents of {dompdb_dir} → {extract_root} and deleted it")
        else:
            logger.warning(f"No 'dompdb' directory found under {extract_root}; skipping flatten step")

        # 3) Ensure every file in extract_root ends in .pdb
        for p in extract_root.iterdir():
            if p.is_file() and p.suffix.lower() != ".pdb":
                newp = p.with_suffix(".pdb")
                # logger.debug(f"Renaming {p.name} → {newp.name}")
                p.rename(newp)

    def _train_test_split(self, threshold: int = 40, seed: int = 1234):
        # 4) Build splits from the .list file
        list_file = self.data_dir / f"cath-dataset-nonredundant-S{threshold}.list"
        with open(list_file) as f:
            all_ids = [line.strip() for line in f if line.strip()]
        
        # Load failed PDBs if the file exists
        failed_pdbs = set()
        failed_file = self.data_dir / "failed_pdbs.txt"
        if failed_file.exists():
            with open(failed_file, "r") as f:
                failed_pdbs = set(line.strip().split('.')[0] for line in f)
            logger.info(f"Loaded {len(failed_pdbs)} failed PDBs to exclude from splits")
        
        # Filter out failed PDBs from the IDs
        filtered_ids = [id for id in all_ids if id not in failed_pdbs]
        logger.info(f"Filtered out {len(all_ids) - len(filtered_ids)} failed PDBs from {len(all_ids)} total IDs")

        random.seed(seed)
        random.shuffle(filtered_ids)
        splits = {
            "validation": filtered_ids[:1000],
            "test":       filtered_ids[1000:2000],
            "train":      filtered_ids[2000:],
        }

        out = self.data_dir / "chain_set_splits.json"
        with open(out, "w") as f:
            json.dump(splits, f)
        logger.info(f"Wrote splits to {out} (seed={seed})")


    def _convert_all_pdbs_to_tensors(self, overwrite: bool = False, n_procs: int = None):
        """
        Parallel convert every .pdb in self.pdb_dir into a graphein Protein tensor
        and save under self.processed_dir as .pt files.
        Shows a tqdm bar with live converted/skipped/failed counts.
        """
        # 1) Prepare
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        pdb_files = sorted(Path(self.pdb_dir).glob("*.pdb"))
        total = len(pdb_files)
        if total == 0:
            logger.warning("No .pdb files found in %s", self.pdb_dir)
            return

        # 2) Build worker args
        args = [
            (str(pdb), str(self.processed_dir), overwrite)
            for pdb in pdb_files
        ]

        converted = skipped = failed = 0
        failed_pdbs = []  # Track failed PDB names

        # 3) Run in parallel with a tqdm wrapper
        with Pool(processes=n_procs) as pool:
            bar = tqdm(
                pool.imap_unordered(_process_single_pdb, args),
                total=total,
                desc="Converting PDBs → .pt",
                unit="file",
                leave=True,
            )
            
            for name, status, err in bar:
                if status == "converted":
                    converted += 1
                    # logger.debug("Saved tensor: %s", name)
                elif status == "skipped":
                    skipped += 1
                    # logger.debug("Skipping existing tensor: %s", name)
                else:  # failed
                    failed += 1
                    failed_pdbs.append(name)  # Add to failed list
                    logger.error(f"Failed to convert {name}: {err}")

                bar.set_postfix(
                    converted=converted,
                    skipped=skipped,
                    failed=failed,
                )

        # Save failed PDBs to a file
        if failed_pdbs:
            failed_file = self.data_dir / "failed_pdbs.txt"
            with open(failed_file, "w") as f:
                for name in failed_pdbs:
                    f.write(f"{name}\n")
            logger.info(f"Saved {len(failed_pdbs)} failed PDB names to {failed_file}")

        # 4) Final summary
        logger.info(
            "Done converting PDBs: total={}, converted={}, skipped={}, failed={}",
            total, converted, skipped, failed
        )

    @functools.lru_cache
    def parse_dataset(self) -> Dict[str, List[str]]:
        """Parses dataset index file

        Returns a dictionary with keys "train", "validation", and "test" and
        values as lists of PDB IDs.

        :return: Dictionary of PDB IDs
        :rtype: Dict[str, List[str]]
        """
        fpath = self.data_dir / "chain_set_splits.json"

        with open(fpath, "r") as file:
            data = json.load(file)

        self.train_pdbs = data["train"]
        logger.info(f"Found {len(self.train_pdbs)} chains in training set")

        logger.info(
            f"Sampling fraction {self.dataset_fraction} of training set"
        )
        fraction = int(self.dataset_fraction * len(self.train_pdbs))
        self.train_pdbs = random.sample(self.train_pdbs, fraction)

        self.val_pdbs = data["validation"]
        logger.info(f"Found {len(self.val_pdbs)} chains in validation set")

        self.test_pdbs = data["test"]
        logger.info(f"Found {len(self.test_pdbs)} chains in test set")
        return data

    def train_dataset(self) -> ProteinDataset:
        """Returns the training dataset.

        :return: Training dataset
        :rtype: ProteinDataset
        """
        if not hasattr(self, "train_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.train_pdbs]

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def val_dataset(self) -> ProteinDataset:
        """Returns the validation dataset.

        :return: Validation dataset
        :rtype: ProteinDataset
        """
        if not hasattr(self, "val_pdbs"):
            self.parse_dataset()

        pdb_codes = [pdb.split(".")[0] for pdb in self.val_pdbs]

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def test_dataset(self) -> ProteinDataset:
        """Returns the test dataset.

        :return: Test dataset
        :rtype: ProteinDataset
        """
        if not hasattr(self, "test_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.test_pdbs]

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def train_dataloader(self) -> ProteinDataLoader:
        """Returns the training dataloader.

        :return: Training dataloader
        :rtype: ProteinDataLoader
        """
        if not hasattr(self, "train_ds"):
            self.train_ds = self.train_dataset()
        return ProteinDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        if not hasattr(self, "val_ds"):
            self.val_ds = self.val_dataset()
        return ProteinDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns the test dataloader.

        :return: Test dataloader
        :rtype: ProteinDataLoader
        """
        if not hasattr(self, "test_ds"):
            self.test_ds = self.test_dataset()
        return ProteinDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )