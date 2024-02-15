from datasets import Dataset
import webdataset as wds
import io

# TODO: Extend below for precomputed features
def convert_hf_dataset2webdataset(dataset: Dataset, image_column: str, caption_column: str, extra_colulmns: tuple[str, ...], tarfile_name: str)
    sink = wds.TarWriter(tarfile_name, encoder=False)
    for i, example in enumerate(dataset):
        image = example[image_column]
        b = io.BytesIO()
        image.save(b, "JPEG")
        image_bytes = b.seek(0).read()
        text = example[caption_column]
        sample = {
            "__key__": str(i),
            image_column: image_bytes,
            caption_column: text
        }
        sink.write(sample)
    sink.close()