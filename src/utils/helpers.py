import os
import urllib.request
import logging

log = logging.getLogger(__name__)

def download_file(url, destination):
    """Downloads a file if it doesn't already exist."""
    if os.path.exists(destination):
        return
    
    log.info(f"Downloading {os.path.basename(destination)}...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        # Simple progress logger
        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size)
            if pct % 10 == 0:
                log.info(f"Download Progress: {pct}%")

        urllib.request.urlretrieve(url, destination)
        log.info(f"Successfully downloaded to {destination}")
    except Exception as e:
        log.error(f"Failed to download {url}: {e}")
        raise
