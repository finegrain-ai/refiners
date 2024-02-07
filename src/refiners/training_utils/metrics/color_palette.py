from typing import Any, Callable, List, Tuple, TypedDict, Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.metrics import ndcg_score  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore

from refiners.training_utils.datasets.color_palette import ColorPalette
from refiners.fluxion.utils import tensor_to_image

from torch import empty, Tensor, cat

Logger = Callable[[Any], None]

class BatchColorPalettePrompt:
    def __init__(
        self,
        source_prompts: List[str],
        source_palettes: List[ColorPalette],
        text_embeddings: Tensor,
        db_indexes: List[int],
        source_images: List[Image.Image]
    ) -> None:
        self.source_prompts = source_prompts
        self.source_palettes = source_palettes
        self.text_embeddings = text_embeddings
        self.db_indexes = db_indexes
        self.source_images = source_images

    @classmethod
    def collate_fn(cls, batch: Sequence["BatchColorPalettePrompt"]) -> "BatchColorPalettePrompt":
        source_palettes = [source_palette for item in batch for source_palette in item.source_palettes]
        source_prompts = [prmpt for item in batch for prmpt in item.source_prompts]
        source_images = [image for item in batch for image in item.source_images]
        return BatchColorPalettePrompt(
            db_indexes=[index for item in batch for index in item.db_indexes],
            source_palettes=source_palettes,
            source_prompts=source_prompts,
            text_embeddings=cat([item.text_embeddings for item in batch]),
            source_images=source_images
        )

class BatchColorPaletteResults(BatchColorPalettePrompt):
    
    result_images: Tensor
    result_palettes: List[ColorPalette]
    
    def __init__(
        self,
        result_images: Tensor,
        result_palettes: List[ColorPalette],
        source_palettes: List[ColorPalette],
        source_prompts: List[str],
        text_embeddings: Tensor,
        source_images: List[Image.Image],
        db_indexes: List[int]
    ) -> None:
        super().__init__(
            source_palettes=source_palettes,
            source_prompts=source_prompts,
            text_embeddings=text_embeddings,
            db_indexes=db_indexes,
            source_images=source_images
        )

        self.result_images = result_images
        self.result_palettes = result_palettes

    def get_prompt(self, prompt: str) -> "BatchColorPaletteResults":
        indices = [i for i, p in enumerate(self.source_prompts) if p == prompt]

        return BatchColorPaletteResults(
            result_images=self.result_images[indices],
            result_palettes=[self.result_palettes[i] for i in indices],
            source_prompts=[prompt for _ in indices],
            source_palettes=[self.source_palettes[i] for i in indices],
            text_embeddings=self.text_embeddings[indices],
            db_indexes=[self.db_indexes[i] for i in indices],
            source_images=[self.source_images[i] for i in indices]
        )

    @classmethod
    def empty(cls) -> "BatchColorPaletteResults":
        return BatchColorPaletteResults(
            result_palettes=[],
            source_palettes=[],
            result_images=empty((0, 3, 512, 512)),
            source_prompts=[],
            text_embeddings=empty((0, 77, 768)),
            db_indexes=[],
            source_images=[]
        )

    def to_prompt(self) -> BatchColorPalettePrompt:
        return BatchColorPalettePrompt(
            source_prompts=self.source_prompts,
            source_palettes=self.source_palettes,
            text_embeddings=self.text_embeddings,
            db_indexes=self.db_indexes,
            source_images=self.source_images
        )

    @classmethod
    def collate_fn(cls, batch: Sequence["BatchColorPaletteResults"]) -> "BatchColorPaletteResults":
        prompts = [item.to_prompt() for item in batch]
        prompt = super().collate_fn(prompts)

        result_images = cat([item.result_images for item in batch])
        result_palettes = [palette for item in batch for palette in item.result_palettes]
        return BatchColorPaletteResults(
            result_images=result_images,
            result_palettes = result_palettes,
            source_prompts=prompt.source_prompts,
            source_palettes=prompt.source_palettes,
            text_embeddings=prompt.text_embeddings,
            db_indexes=prompt.db_indexes,
            source_images=prompt.source_images
        )


class BatchHistogramPrompt:
    def __init__(
        self,
        source_histogram_embeddings: Tensor,
        source_histograms: Tensor,
        source_prompts: List[str],
        palettes: List[ColorPalette],
        text_embeddings: Tensor,
        db_indexes: List[int],
        source_images: List[Image.Image]
    ) -> None:
        self.source_histogram_embeddings = source_histogram_embeddings
        self.source_histograms = source_histograms
        self.source_prompts = source_prompts
        self.palettes = palettes
        self.text_embeddings = text_embeddings
        self.db_indexes = db_indexes
        self.source_images = source_images

    @classmethod
    def collate_fn(cls, batch: Sequence["BatchHistogramPrompt"]) -> "BatchHistogramPrompt":
        source_histograms = cat([item.source_histograms for item in batch])
        source_histogram_embeddings = cat([item.source_histogram_embeddings for item in batch])
        palettes = [palette for item in batch for palette in item.palettes]
        source_prompts = [prmpt for item in batch for prmpt in item.source_prompts]
        source_images = [image for item in batch for image in item.source_images]
        return BatchHistogramPrompt(
            db_indexes=[index for item in batch for index in item.db_indexes],
            source_histograms=source_histograms,
            source_histogram_embeddings=source_histogram_embeddings,
            source_prompts=source_prompts,
            palettes=palettes,
            text_embeddings=cat([item.text_embeddings for item in batch]),
            source_images=source_images
        )


class BatchHistogramResults(BatchHistogramPrompt):
    result_images: Tensor
    result_histograms: Tensor

    def __init__(
        self,
        result_images: Tensor,
        result_histograms: Tensor,
        source_histogram_embeddings: Tensor,
        source_histograms: Tensor,
        source_prompts: List[str],
        palettes: List[ColorPalette],
        text_embeddings: Tensor,
        source_images: List[Image.Image],
        db_indexes: List[int]
    ) -> None:
        super().__init__(
            source_histogram_embeddings=source_histogram_embeddings,
            source_histograms=source_histograms,
            source_prompts=source_prompts,
            palettes=palettes,
            text_embeddings=text_embeddings,
            db_indexes=db_indexes,
            source_images=source_images
        )

        self.result_images = result_images
        self.result_histograms = result_histograms

    def get_prompt(self, prompt: str) -> "BatchHistogramResults":
        indices = [i for i, p in enumerate(self.source_prompts) if p == prompt]

        return BatchHistogramResults(
            result_images=self.result_images[indices],
            result_histograms=self.result_histograms[indices],
            source_histogram_embeddings=self.source_histogram_embeddings[indices],
            source_histograms=self.source_histograms[indices],
            source_prompts=[prompt for _ in indices],
            palettes=[self.palettes[i] for i in indices],
            text_embeddings=self.text_embeddings[indices],
            db_indexes=[self.db_indexes[i] for i in indices],
            source_images=[self.source_images[i] for i in indices]
        )

    @classmethod
    def empty(cls) -> "BatchHistogramResults":
        return BatchHistogramResults(
            result_images=empty((0, 3, 512, 512)),
            result_histograms=empty((0, 64, 64, 64)),
            source_histogram_embeddings=empty((0, 8, 2, 2, 2)),
            source_histograms=empty((0, 64, 64, 64)),
            source_prompts=[],
            palettes=[],
            text_embeddings=empty((0, 77, 768)),
            db_indexes=[],
            source_images=[]
        )

    def to_hist_prompt(self) -> BatchHistogramPrompt:
        return BatchHistogramPrompt(
            source_histogram_embeddings=self.source_histogram_embeddings,
            source_histograms=self.source_histograms,
            source_prompts=self.source_prompts,
            palettes=self.palettes,
            text_embeddings=self.text_embeddings,
            db_indexes=self.db_indexes,
            source_images=self.source_images
        )

    @classmethod
    def collate_fn(cls, batch: Sequence["BatchHistogramResults"]) -> "BatchHistogramResults":
        histo_prompts = [item.to_hist_prompt() for item in batch]
        histo_prompt = super().collate_fn(histo_prompts)

        result_images = cat([item.result_images for item in batch])
        result_histograms = cat([item.result_histograms for item in batch])
        return BatchHistogramResults(
            result_images=result_images,
            result_histograms=result_histograms,
            source_histograms=histo_prompt.source_histograms,
            source_histogram_embeddings=histo_prompt.source_histogram_embeddings,
            source_prompts=histo_prompt.source_prompts,
            palettes=histo_prompt.palettes,
            text_embeddings=histo_prompt.text_embeddings,
            db_indexes=histo_prompt.db_indexes,
            source_images=histo_prompt.source_images
        )



class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: ColorPalette


def image_palette_metrics(
    image: Image.Image, palette: ColorPalette, img_size: Tuple[int, int] = (256, 256), sampling_size: int = 1000
):
    resized_img = image.resize(img_size)
    Point = npt.NDArray[np.float64]
    all_points: List[Point] = np.array(resized_img.getdata(), dtype=np.float64)  # type: ignore
    choices = np.random.choice(len(all_points), sampling_size)
    points = all_points[choices]

    num = len(palette)

    centroids = np.stack(palette)

    nn = NearestNeighbors(n_neighbors=num)
    nn.fit(centroids)  # type: ignore

    indices: npt.NDArray[np.int8] = nn.kneighbors(points, return_distance=False)  # type: ignore
    indices = indices[:, 0]

    counts = np.bincount(indices)  # type: ignore
    counts = np.pad(counts, (0, num - len(counts)), "constant")  # type: ignore
    y_true_ranking = list(range(num, 0, -1))

    distances_list: List[float] = []

    def distance(a: Point, b: Point) -> float:
        return np.linalg.norm(a - b).item()

    for i in range(len(centroids)):
        condition = np.where(indices == i)

        cluster_points = points[condition]
        distances = [distance(p, centroids[i]) for p in cluster_points]
        distances_list.extend(distances)

    return ([y_true_ranking], [counts], distances_list)


def batch_image_palette_metrics(log: Logger, images_and_palettes: List[ImageAndPalette], prefix: str = "palette-img"):
    per_num: dict[int, Any] = {}
    for image_and_palette in images_and_palettes:
        palette = image_and_palette["palette"]
        image = image_and_palette["image"]
        num = len(palette)

        (y_true_ranking, counts, distances_list) = image_palette_metrics(image, palette)
        if not num in per_num:
            per_num[num] = {
                "y_true_ranking": y_true_ranking,
                "counts": counts,
                "distances": distances_list,
            }
        else:
            per_num[num]["y_true_ranking"] += y_true_ranking
            per_num[num]["counts"] += counts
            per_num[num]["distances"] += distances_list

    for num in per_num:
        if num > 1:
            score: float = ndcg_score(per_num[num]["y_true_ranking"], per_num[num]["counts"]).item()
            log({f"{prefix}/ndcg_{num}": score, f"{prefix}/std_dev_{num}": np.std(per_num[num]["distances"]).item()})
        else:
            log({f"{prefix}/std_dev_{num}": np.std(per_num[num]["distances"]).item()})

def batch_palette_metrics(log: Logger, images_and_palettes: BatchHistogramResults, prefix: str = "palette-img"):
    
    images = [tensor_to_image(image) for image in images_and_palettes.result_images.split(1)]
    palettes = images_and_palettes.palettes
    
    if len(images) != len(palettes):
        raise ValueError("Images and palettes must have the same length")
    
    return batch_image_palette_metrics(
        log, 
        [{"image": image, "palette": palette} for image, palette in zip(images, palettes)], 
        prefix
    )