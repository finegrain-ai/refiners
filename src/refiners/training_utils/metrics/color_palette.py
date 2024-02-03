from typing import Any, Callable, List, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.metrics import ndcg_score  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore

from refiners.training_utils.datasets.color_palette import ColorPalette
from src.refiners.fluxion.utils import tensor_to_image
from src.refiners.training_utils.trainers.histogram import BatchHistogramResults

Logger = Callable[[Any], None]


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
    
    images = [tensor_to_image(image) for image in images_and_palettes["images"].split(1)]
    palettes = images_and_palettes["palettes"]
    
    if len(images) != len(palettes):
        raise ValueError("Images and palettes must have the same length")
    
    return batch_image_palette_metrics(
        log, 
        [{"image": image, "palette": palette} for image, palette in zip(images, palettes)], 
        prefix
    )