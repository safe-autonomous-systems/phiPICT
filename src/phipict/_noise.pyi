from __future__ import annotations
import collections.abc
import torch
import typing
__all__: list[str] = ['CURL', 'CURL_FBM', 'FRACTAL_BROWNIAN_MOTION', 'GRADIENT', 'GRADIENT_FBM', 'GenerateSimplexNoiseVariation', 'NoiseVariation', 'RIDGED', 'RIDGED_MULTI_FRACTAL', 'SIMPLEX', 'WORLEY']
class NoiseVariation:
    """
    Members:
    
      SIMPLEX
    
      WORLEY
    
      FRACTAL_BROWNIAN_MOTION
    
      RIDGED
    
      RIDGED_MULTI_FRACTAL
    
      GRADIENT
    
      GRADIENT_FBM
    
      CURL
    
      CURL_FBM
    """
    CURL: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.CURL: 7>
    CURL_FBM: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.CURL_FBM: 8>
    FRACTAL_BROWNIAN_MOTION: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.FRACTAL_BROWNIAN_MOTION: 2>
    GRADIENT: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.GRADIENT: 5>
    GRADIENT_FBM: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.GRADIENT_FBM: 6>
    RIDGED: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.RIDGED: 3>
    RIDGED_MULTI_FRACTAL: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.RIDGED_MULTI_FRACTAL: 4>
    SIMPLEX: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.SIMPLEX: 0>
    WORLEY: typing.ClassVar[NoiseVariation]  # value = <NoiseVariation.WORLEY: 1>
    __members__: typing.ClassVar[dict[str, NoiseVariation]]  # value = {'SIMPLEX': <NoiseVariation.SIMPLEX: 0>, 'WORLEY': <NoiseVariation.WORLEY: 1>, 'FRACTAL_BROWNIAN_MOTION': <NoiseVariation.FRACTAL_BROWNIAN_MOTION: 2>, 'RIDGED': <NoiseVariation.RIDGED: 3>, 'RIDGED_MULTI_FRACTAL': <NoiseVariation.RIDGED_MULTI_FRACTAL: 4>, 'GRADIENT': <NoiseVariation.GRADIENT: 5>, 'GRADIENT_FBM': <NoiseVariation.GRADIENT_FBM: 6>, 'CURL': <NoiseVariation.CURL: 7>, 'CURL_FBM': <NoiseVariation.CURL_FBM: 8>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def GenerateSimplexNoiseVariation(output_shape: collections.abc.Sequence[typing.SupportsInt], GPUdevice: torch.device, scale: collections.abc.Sequence[typing.SupportsFloat], offset: collections.abc.Sequence[typing.SupportsFloat], variation: NoiseVariation, ridgeOffset: typing.SupportsFloat = 1.0, octaves: typing.SupportsInt = 4, lacunarity: typing.SupportsFloat = 2.0, gain: typing.SupportsFloat = 0.5) -> torch.Tensor:
    """
    Create a tensor with simplex noise.
    """
CURL: NoiseVariation  # value = <NoiseVariation.CURL: 7>
CURL_FBM: NoiseVariation  # value = <NoiseVariation.CURL_FBM: 8>
FRACTAL_BROWNIAN_MOTION: NoiseVariation  # value = <NoiseVariation.FRACTAL_BROWNIAN_MOTION: 2>
GRADIENT: NoiseVariation  # value = <NoiseVariation.GRADIENT: 5>
GRADIENT_FBM: NoiseVariation  # value = <NoiseVariation.GRADIENT_FBM: 6>
RIDGED: NoiseVariation  # value = <NoiseVariation.RIDGED: 3>
RIDGED_MULTI_FRACTAL: NoiseVariation  # value = <NoiseVariation.RIDGED_MULTI_FRACTAL: 4>
SIMPLEX: NoiseVariation  # value = <NoiseVariation.SIMPLEX: 0>
WORLEY: NoiseVariation  # value = <NoiseVariation.WORLEY: 1>
