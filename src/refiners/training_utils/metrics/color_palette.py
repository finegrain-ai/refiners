from typing import Any, Callable, List, Tuple, TypeVar, TypedDict, Sequence, Type, Generic, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.metrics import ndcg_score  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore

from refiners.fluxion.utils import tensor_to_images
from refiners.fluxion.adapters.color_palette import Color, ColorPalette

from torch import empty, Tensor, cat

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: list[Color]
    
Logger = Callable[[Any], None]

CollatableProps = list[Any] | Tensor

PromptType = TypeVar('PromptType', bound='AbstractColorPrompt')

class AbstractColorPrompt:
    _list_keys: list[str] = []
    _tensor_keys: dict[str, tuple[int, ...]] = {}
    
    def __init__(
        self,
        **kwargs : CollatableProps
    ) -> None:
        for key in self.__class__._list_keys:
            if key not in kwargs:
                raise ValueError(f"Key {key} is not present in {kwargs}")
            setattr(self, key, kwargs[key])
        for key in self.__class__._tensor_keys:
            if key not in kwargs:
                raise ValueError(f"Key {key} is not present in {kwargs}")
            setattr(self, key, kwargs[key])

    @classmethod
    def collate_fn(cls: Type[PromptType], batch: Sequence["AbstractColorPrompt"]) -> PromptType:
        opts : dict[str, CollatableProps] = {}
        for key in cls._list_keys:
            opts[key] : list[Any] = []

            for item in batch:
                if not hasattr(item, key):
                    raise ValueError(f"Key {key} is not present in {item}")
                for prop in getattr(item, key):
                    opts[key].append(prop)
        for key in cls._tensor_keys:
            lst : list[Tensor] = []
            for item in batch:
                if not hasattr(item, key):
                    raise ValueError(f"Key {key} is not present in {item}")
                tensor = getattr(item, key)
                if not isinstance(tensor, Tensor):
                    raise ValueError(f"Key {key}, {tensor} should be a tensor")
                lst.append(tensor)
            opts[key] = cat(lst)
            
        return cls(**opts)
    
    @classmethod
    def empty(cls: Type[PromptType]) -> PromptType:
        opts : dict[str, CollatableProps] = {}
        
        for key in cls._list_keys:
            opts[key] = []
        for key in cls._tensor_keys:
            size = cls._tensor_keys[key]
            tensor = empty((0,)+ size)
            opts[key] = tensor
            
        return cls(**opts)

    def get_indices(self: PromptType, indices: list[int]) -> PromptType:
        opts : dict[str, CollatableProps] = {}
        
        for key in self.__class__._list_keys:
            opts[key] = [getattr(self, key)[i] for i in indices]
        for key in self._tensor_keys:
            opts[key] = getattr(self, key)[indices]
            
        return self.__class__(**opts)
    
    def get_prompt(self: PromptType, prompt: str) -> PromptType:
        prompts = cast(list[str], getattr(self, "source_prompts"))
        indices = [i for i, p in enumerate(prompts) if p == prompt]
        return self.get_indices(indices)
    
class AbstractColorResults(Generic[PromptType], AbstractColorPrompt):
    __prompt_type: Type[PromptType]
    
    def to_prompt(self) -> PromptType:
        opts : dict[str, CollatableProps] = {}
        
        for key in self.__prompt_type._list_keys:
            opts[key] = getattr(self, key)
        for key in self.__prompt_type._tensor_keys:
            opts[key] = getattr(self, key)
        
        return self.__prompt_type()



class BatchColorPalettePrompt(AbstractColorPrompt):
    _list_keys: List[str] = ["source_palettes", "source_prompts", "source_images", "db_indexes"]
    _tensor_keys: dict[str, tuple[int, ...]] = {
        "text_embeddings": (77, 768)
    }

class BatchColorPaletteResults(AbstractColorResults[BatchColorPalettePrompt]):    
    _list_keys: List[str] = ["source_palettes", "source_prompts", "source_images", "db_indexes", "result_palettes"]
    _tensor_keys: dict[str, tuple[int, ...]] = {
        "text_embeddings": (77, 768),
        "result_images": (3, 512, 512)
    }

class BatchHistogramPrompt(AbstractColorPrompt):
    _list_keys: List[str] = ["source_palettes", "source_prompts", "source_images", "db_indexes"]
    _tensor_keys: dict[str, tuple[int, ...]] = {
        "source_histogram_embeddings": (8, 2, 2, 2),
        "source_histograms": (64, 64, 64),
        "text_embeddings": (77, 768),
    }

class BatchHistogramResults(AbstractColorResults[AbstractColorPrompt]):
    _list_keys: List[str] = ["source_palettes", "source_prompts", "source_images", "db_indexes"]
    _tensor_keys: dict[str, tuple[int, ...]] = {
        "source_histogram_embeddings": (8, 2, 2, 2),
        "source_histograms": (64, 64, 64),
        "text_embeddings": (77, 768),
        "result_images": (3, 512, 512),
        "result_histograms": (64, 64, 64)
    }

def image_palette_metrics(
    image: Image.Image, palette: list[Color], img_size: Tuple[int, int] = (256, 256), sampling_size: int = 1000
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


def batch_image_palette_metrics(log: Logger, images_and_palettes: list[ImageAndPalette], prefix: str = "palette-img"):
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
    
    source_palettes = cast(list[ColorPalette], images_and_palettes.source_palettes) # type: ignore
    palettes: list[list[Color]] = []
    for p in source_palettes: # type: ignore
        colors : list[Color] = []
        sorted_clusters = sorted(p, key=lambda x: x[1], reverse=True)
        for sorted_clusters in p:
            colors.append(sorted_clusters[0])
        palettes.append(colors)
    
    images = tensor_to_images(images_and_palettes.result_images) # type: ignore
    
    if len(images) != len(palettes):
        raise ValueError("Images and palettes must have the same length")
        
    return batch_image_palette_metrics(
        log,
        [
            ImageAndPalette({"image": image, "palette": palette})
            for image, palette in zip(images, palettes)
        ],
        prefix
    )