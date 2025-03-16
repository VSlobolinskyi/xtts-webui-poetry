import os
import subprocess
import sys
import time
import concurrent
import aria2p
import requests

import importlib.metadata as metadata  # Use importlib.metadata
from pathlib import Path
from tqdm import tqdm

from packaging import version
from loguru import logger


def install_package(package_link):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package_link])


def is_package_installed(package_name):
    try:
        metadata.version(package_name)
        return True
    except metadata.PackageNotFoundError:
        return False


def check_and_install_torch():
    required_torch_version = 'torch+cu118'

    # Check if torch with CUDA 11.8 is installed.
    if not any(required_torch_version in pkg for pkg in metadata.distributions()):
        logger.info(f"'{required_torch_version}' not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", required_torch_version])
    else:
        logger.info(f"'{required_torch_version}' already installed.")


def install_deepspeed_based_on_python_version():
    # check_and_install_torch()
    if not is_package_installed('deepspeed'):
        python_version = sys.version_info

        logger.info(
            f"Python version: {python_version.major}.{python_version.minor}")

        py311_win = "https://github.com/daswer123/xtts-webui/releases/download/deepspeed/deepspeed-0.11.2+cuda118-cp311-cp311-win_amd64.whl"

        # Use generic pip install deepspeed for Linux or custom wheels for Windows.
        deepspeed_link = None

        if sys.platform == 'win32':
            deepspeed_link = py311_win
        else:
            deepspeed_link = 'deepspeed==0.11.2'

        if deepspeed_link:
            logger.info("Installing DeepSpeed...")
            install_package(deepspeed_link)
        else:
            logger.info("'deepspeed' already installed.")


def create_directory_if_not_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True)


def init_aria2():
    try:
        aria2 = aria2p.API(
            aria2p.Client(
                host="http://localhost",
                port=6800,
                secret=""  # If you set a secret token, put it here
            )
        )
        # Test connection
        aria2.get_global_stat()
        return aria2
    except Exception as e:
        print(f"[XTTS] Could not connect to aria2 RPC server: {e}")
        print("[XTTS] Falling back to aria2c command line")
        return None

def download_file(url, destination):
    """Download smaller files with regular requests"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8 KiB blocks
    
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"[XTTS] {destination.name}")
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def download_large_file_aria2(url, destination):
    """Download large files using aria2 for maximum speed"""
    dest_dir = os.path.dirname(destination)
    dest_file = os.path.basename(destination)
    
    # Convert Path object to string if needed
    if isinstance(destination, Path):
        dest_dir = str(destination.parent)
        dest_file = str(destination.name)
    
    # Try to use aria2p API first
    aria2 = init_aria2()
    
    if aria2:
        # Use the aria2p library to download
        try:
            print(f"[XTTS] Starting download of {dest_file} using aria2 RPC...")
            
            # Add download to aria2
            download = aria2.add_uris(
                [url],
                options={
                    "dir": dest_dir,
                    "out": dest_file,
                    "continue": True,  # Resume download if possible
                    "max-connection-per-server": 16,  # Multiple connections per server
                    "split": 16,  # Split file into 16 parts
                    "min-split-size": "1M",  # Minimum 1MB per split
                    "max-concurrent-downloads": 1,
                    "file-allocation": "none",  # Faster file allocation
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            
            # Monitor download progress
            with tqdm(total=1, unit='%', desc=f"[XTTS] {dest_file}", position=0) as progress_bar:
                while download.is_active:
                    download.update()
                    progress = download.progress / 100 if download.total_length > 0 else 0
                    progress_bar.n = progress
                    progress_bar.refresh()
                    speed = download.download_speed_string()
                    progress_bar.set_postfix(speed=speed)
                    time.sleep(0.5)
            
            # Check if download was successful
            if download.status == 'complete':
                print(f"[XTTS] Download of {dest_file} completed successfully")
                return
            else:
                print(f"[XTTS] Download failed with status: {download.status}")
                # Fall back to command line if RPC fails
        except Exception as e:
            print(f"[XTTS] Error using aria2 RPC: {e}")
            print("[XTTS] Falling back to aria2c command line")
    
    # If aria2p fails or is not available, use command line
    try:
        # Build command for aria2c
        cmd = [
            "aria2c",
            "--dir", dest_dir,
            "--out", dest_file,
            "--file-allocation=none",
            "--continue=true",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            url
        ]
        
        print(f"[XTTS] Starting download of {dest_file} using aria2c command line...")
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor download progress
        for line in process.stdout:
            # Parse progress from aria2c output
            if '%' in line:
                try:
                    # Extract percentage and speed from aria2c output
                    percentage_part = line.split('[')[1].split(']')[0] if '[' in line and ']' in line else ''
                    percentage = float(percentage_part.strip('%')) / 100 if percentage_part else 0
                    speed = line.split('DL:')[1].split('/s')[0].strip() + '/s' if 'DL:' in line and '/s' in line else 'N/A'
                    
                    # Clear line and print progress
                    print(f"\r[XTTS] {dest_file}: {percentage:.1%} complete, Speed: {speed}", end='')
                except:
                    # If parsing fails, just print the line
                    print(line.strip())
            else:
                print(line.strip())
        
        # Wait for process to complete
        process.wait()
        print()  # New line after progress display
        
        # Check if download was successful
        if process.returncode == 0:
            print(f"[XTTS] Download of {dest_file} completed successfully")
        else:
            raise Exception(f"aria2c exited with code {process.returncode}")
    except Exception as e:
        print(f"[XTTS] Error using aria2c command: {e}")
        print("[XTTS] Falling back to regular download method")
        
        # Fall back to regular download method if aria2c fails
        fallback_download_large_file(url, destination)

def fallback_download_large_file(url, destination):
    """Fallback method if aria2 is not available"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Check if partial file exists to enable resume
    file_size = 0
    if destination.exists():
        file_size = os.path.getsize(destination)
        if file_size > 0:
            session.headers.update({'Range': f'bytes={file_size}-'})
    
    # Timeout handling
    max_retries = 5
    retry_count = 0
    chunk_size = 1024 * 1024 * 4  # 4 MiB chunks
    mode = 'ab' if file_size > 0 else 'wb'
    
    while retry_count < max_retries:
        try:
            response = session.get(url, stream=True, timeout=30)
            
            # Handle resume or normal download
            total_size = int(response.headers.get('content-length', 0))
            if file_size > 0 and response.status_code == 206:
                total_size = total_size + file_size
                print(f"[XTTS] Resuming download from {file_size/(1024*1024):.2f} MB")
            elif file_size > 0 and response.status_code == 200:
                print("[XTTS] Server doesn't support resume, starting from beginning")
                total_size = int(response.headers.get('content-length', 0))
                file_size = 0
                mode = 'wb'
            
            # Show progress
            progress_bar = tqdm(
                initial=file_size,
                total=total_size,
                unit='iB', 
                unit_scale=True,
                desc=f"[XTTS] {destination.name}"
            )
            
            # Download the file
            with open(destination, mode) as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            return
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            retry_count += 1
            wait_time = retry_count * 5
            print(f"[XTTS] Connection error: {e}. Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
            
            # Update file size for resuming
            if destination.exists():
                file_size = os.path.getsize(destination)
                if file_size > 0:
                    session.headers.update({'Range': f'bytes={file_size}-'})
    
    raise Exception(f"Failed to download {destination.name} after {max_retries} attempts")


def upgrade_tts_package():
    try:
        logger.warning("TTS version is outdated, attempting to upgrade TTS...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "tts"])
        logger.info("TTS has been successfully upgraded ")
    except Exception as e:
        logger.error(f"An error occurred while upgrading TTS: {e}")
        logger.info("Try installing the new version manually")
        logger.info("pip install --upgrade tts")


def upgrade_stream2sentence_package():
    try:
        logger.warning(
            "Stream2sentence version is outdated, attempting to upgrade stream2sentence...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "stream2sentence"])
        logger.info("Stream2sentence has been successfully upgraded ")
    except Exception as e:
        logger.error(f"An error occurred while upgrading Stream2sentence: {e}")
        logger.info("Stream2sentence installing the new version manually")
        logger.info("pip install --upgrade stream2sentence")


def check_tts_version():
    try:
        tts_version = metadata.version("tts")

        if version.parse(tts_version) < version.parse("0.21.2"):
            upgrade_tts_package()

    except metadata.PackageNotFoundError:
        print("TTS is not installed.")


def check_stream2sentence_version():
    try:
        tts_version = metadata.version("stream2sentence")
        if version.parse(tts_version) < version.parse("0.2.2"):
            upgrade_stream2sentence_package()
    except metadata.PackageNotFoundError:
        print("stream2sentence is not installed.")


def get_folder_names(directory):
    # Make sure that the given directory is indeed a directory
    if not os.path.isdir(directory):
        raise ValueError(
            f"The provided path '{directory}' is not a valid directory.")

    # List all entries in the given directory and filter out files, keeping only directories
    folder_names = [name for name in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, name))]

    return folder_names


def get_folder_names_advanced(directory):
    # Список специфических названий папок для добавления при необходимости
    specific_folders = ["v2.0.3", "v2.0.2", "main"]

    # Убедиться, что предоставленный путь является действительной директорией
    if not os.path.isdir(directory):
        raise ValueError(
            f"The provided path '{directory}' is not a valid directory.")

    # Получить список всех подпапок в данной директории
    folder_names = [name for name in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, name))]

    # Добавить отсутствующие специфические папки в начало списка folder_names
    folders_to_add = [
        folder for folder in specific_folders if folder not in folder_names]

    return folders_to_add + folder_names


def download_model(this_dir, model_version):
    """Download model files using aria2 for large files"""
    # Define paths
    base_path = this_dir / 'models'
    model_path = base_path / f'{model_version}'

    # Define files and their corresponding URLs
    files_to_download = {
        "config.json": f"https://huggingface.co/coqui/XTTS-v2/raw/{model_version}/config.json",
        "model.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/model.pth?download=true",
        "vocab.json": f"https://huggingface.co/coqui/XTTS-v2/raw/{model_version}/vocab.json",
        "speakers_xtts.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/main/speakers_xtts.pth?download=true"
    }

    # Check and create directories
    create_directory_if_not_exists(base_path)
    create_directory_if_not_exists(model_path)

    # Download files using a thread pool for small files and aria2 for large ones
    large_files = ["model.pth", "speakers_xtts.pth"]
    small_files = [f for f in files_to_download.keys() if f not in large_files]
    
    # Download small files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(download_file, files_to_download[filename], model_path / filename): filename
            for filename in small_files if not (model_path / filename).exists()
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                future.result()
                print(f"[XTTS] Downloaded {filename}")
            except Exception as exc:
                print(f"[XTTS] Error downloading {filename}: {exc}")
    
    # Download large files with aria2
    for filename in large_files:
        destination = model_path / filename
        if not destination.exists():
            print(f"[XTTS] Downloading {filename}...")
            download_large_file_aria2(files_to_download[filename], destination)


if __name__ == "__main__":
    #    this_dir = Path(__file__).parent.resolve()
    #    main_downloader(this_dir)
    install_deepspeed_based_on_python_version()
