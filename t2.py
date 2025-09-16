import time
from datasets import load_dataset, DownloadConfig

def slow_download_dataset(split="train", batch_size=100, delay=10, max_retries=5):
    total_downloaded = 0
    dataset_size = 65550  # Approximate dataset size (adjust if known)
    full_data = None

    while total_downloaded < dataset_size:
        start = total_downloaded
        end = min(start + batch_size, dataset_size)

        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                print(f"Downloading samples {start} to {end}...")
                batch = load_dataset(
                    "CGIAR/gardian-cigi-ai-documents",
                    split=f"{split}[{start}:{end}]",
                    download_config=DownloadConfig(max_retries=max_retries)
                )

                if len(batch) == 0:
                    print("No more data to download.")
                    break

                if full_data is None:
                    full_data = batch
                else:
                    full_data = full_data.concatenate(batch)

                total_downloaded += len(batch)
                print(f"Total downloaded: {total_downloaded} samples")

                success = True
                time.sleep(delay)  # Slow down to avoid rate limit

            except Exception as e:
                if "429" in str(e):
                    wait_time = 2 ** retry_count
                    print(f"Rate limited! Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    print(f"Error: {e}")
                    raise

        if not success:
            print("Failed after max retries, stopping download.")
            break

    if full_data:
        save_path = "gardian_cigi_ai_documents_full.jsonl"
        full_data.to_json(save_path)
        print(f"Dataset saved locally to {save_path}")

if __name__ == "__main__":
    slow_download_dataset()
