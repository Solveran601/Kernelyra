"""Built-in, data-driven format catalogue.

Recognition, extraction and training are deliberately separate capabilities.
The catalogue never claims that recognizing an extension means the current
training core can learn from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FormatDescriptor:
    id: str
    extension: str
    category: str
    modality: str
    role: str
    handler: str
    training: str
    streaming: bool
    dependency: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Every extension is an explicit built-in route. Formats that share a safe
# parser strategy intentionally share a handler; this avoids 200 copies of the
# same parsing code while preserving an individual descriptor for every route.
_GROUPS: tuple[tuple[str, str, str, str, str, bool, str | None, str], ...] = (
    (
        "plain-text", "text", "dataset", "text-stream", "extract", True, None,
        "txt text md markdown mdown mkd mdx rst rest adoc asciidoc org tex latex bib log nfo readme me wiki textile creole csvs tsvs corpus prompt prompts completion completions instruction instructions transcript transcripts subtitle subtitles srt vtt ass ssa lrc"
    ),
    (
        "source-code", "text", "dataset", "source-text", "extract", True, None,
        "py pyw pyx pxd pyi c h cc cpp cxx c++ hpp hxx hh inl m mm rs go java kt kts scala sc groovy gradle cs fs fsx vb swift dart js jsx mjs cjs ts tsx php phtml rb rake gemspec sh bash zsh fish ps1 psm1 psd1 bat cmd lua r jl zig f f77 f90 f95 f03 f08 for ftn asm s nasm masm v vhd vhdl sv svh cl cu cuh metal sol move ex exs erl hrl hs lhs ml mli clj cljs cljc edn lisp lsp scm ss rkt tcl awk sed cob cbl pas pp d nim crystal cr vala"
    ),
    (
        "structured-text", "table", "dataset", "structured-reader", "extract", True, None,
        "json json5 jsonc jsonld geojson topojson yaml yml toml ini cfg conf config properties env dotenv xml xsd xsl xslt dtd html htm xhtml shtml css scss sass less csv tsv psv ssv jsonl ndjson arff libsvm svmlight sql graphql gql proto thrift rss atom plist manifest lock ipynb notebook nbib ris csl"
    ),
    (
        "office-document", "text", "dataset", "document-extractor", "extract", False, "document-extra",
        "pdf rtf doc docx docm dot dotx odt ott fodt pages wpd sxw abw epub mobi azw azw3 fb2 djvu chm ppt pptx pptm odp fodp key",
    ),
    (
        "spreadsheet", "table", "dataset", "spreadsheet-reader", "extract", True, "data-extra",
        "xls xlsx xlsm xlsb xlt xltx ods ots fods numbers dif sylk slk sav zsav por sas7bdat xpt dta",
    ),
    (
        "columnar-array", "table", "dataset", "tabular-reader", "extract", True, "data-extra",
        "parquet pq feather arrow ipc orc avro npy npz mat h5 hdf5 hdf hd5 zarr fst",
    ),
    (
        "database", "table", "dataset", "database-reader", "extract", True, "database-extra",
        "sqlite sqlite3 db db3 duckdb mdb accdb dbf odb realm rocksdb leveldb",
    ),
    (
        "raster-image", "image", "dataset", "image-decoder", "extract", True, "vision-extra",
        "png jpg jpeg jpe jfif webp gif bmp dib tif tiff heic heif avif jxl jp2 j2k jpf jpx ico cur tga pcx ppm pgm pbm pnm xbm xpm psd psb dds exr hdr pic ras qoi",
    ),
    (
        "camera-raw", "image", "dataset", "raw-image-decoder", "extract", True, "vision-extra",
        "raw dng cr2 cr3 nef nrw arw srf sr2 raf orf rw2 pef x3f erf kdc dcr mrw mos",
    ),
    (
        "vector-image", "image", "dataset", "vector-decoder", "extract", True, "vision-extra",
        "svg svgz ai eps ps pdfa cdr emf wmf vsd vsdx sk sketch fig",
    ),
    (
        "audio", "audio", "dataset", "audio-decoder", "extract", True, "audio-extra",
        "wav wave mp3 flac ogg oga opus m4a aac ac3 eac3 wma aiff aif aifc alac amr ape au snd voc caf mka mid midi kar mod xm it s3m",
    ),
    (
        "video", "video", "dataset", "media-demux", "extract", True, "video-extra",
        "mp4 m4v mkv mov qt avi webm mpg mpeg mpe m1v m2v mts m2ts ts vob wmv asf flv f4v 3gp 3g2 ogv rm rmvb divx h264 h265 hevc y4m mxf",
    ),
    (
        "scene-3d", "3d", "dataset", "scene-parser", "extract", True, "3d-extra",
        "obj fbx gltf glb stl ply dae 3ds blend usd usda usdc usdz abc alembic step stp stpz iges igs dxf dwg brep off 3mf x3d x3db x3dv vrml wrl max ma mb c4d lwo lws scad amf 3dm skp assimp bvh md2 md3 md5mesh nif smd vta",
    ),
    (
        "point-cloud", "3d", "dataset", "point-cloud-reader", "extract", True, "3d-extra",
        "pcd las laz e57 pts ptx xyz xyzn xyzrgb asc copc",
    ),
    (
        "geospatial", "mixed", "dataset", "geospatial-reader", "extract", True, "geo-extra",
        "shp shx prj kml kmz gpx geotiff gpkg mbtiles osm pbf grib grb netcdf nc",
    ),
    (
        "archive", "mixed", "container", "bounded-archive-reader", "extract", True, "archive-extra",
        "zip zipx tar tgz tbz tbz2 txz gz gzip bz2 bzip2 xz lz lz4 lzh lha zst zstd 7z rar cab ar cpio deb rpm apk war jar whl",
    ),
    (
        "model", "model", "model", "model-container", "inspect", False, None,
        "gguf ggml safetensors pt pth pkl pickle joblib ckpt keras onnx pb tflite torchscript mlmodel mlpackage coreml xgb cbm pmml h2o mar params weights engine plan trt"
    ),
)


def _build_catalogue() -> tuple[FormatDescriptor, ...]:
    result: list[FormatDescriptor] = []
    seen: set[str] = set()
    directly_trainable = {"csv", "tsv", "jsonl", "ndjson", "parquet", "pq", "npz"}
    for category, modality, role, handler, default_training, streaming, dependency, raw_extensions in _GROUPS:
        for extension in raw_extensions.split():
            normalized = extension.lower().lstrip(".")
            if normalized in seen:
                continue
            seen.add(normalized)
            training = "train" if normalized in directly_trainable else (
                default_training if dependency is None else "recognize"
            )
            result.append(
                FormatDescriptor(
                    id=normalized,
                    extension=f".{normalized}",
                    category=category,
                    modality=modality,
                    role=role,
                    handler=handler,
                    training=training,
                    streaming=streaming,
                    dependency=dependency,
                )
            )
    return tuple(sorted(result, key=lambda item: item.id))


BUILTIN_FORMATS = _build_catalogue()
FORMAT_BY_EXTENSION = {item.extension: item for item in BUILTIN_FORMATS}
FORMAT_COUNT = len(BUILTIN_FORMATS)

if FORMAT_COUNT < 180:  # pragma: no cover - import-time invariant
    raise RuntimeError(f"Built-in format catalogue regressed below 180 routes: {FORMAT_COUNT}")


def describe_formats() -> list[dict[str, Any]]:
    return [item.to_dict() for item in BUILTIN_FORMATS]


def format_counts() -> dict[str, int]:
    return {
        "recognized": FORMAT_COUNT,
        "extractable": sum(item.training in {"extract", "train"} for item in BUILTIN_FORMATS),
        "directly_trainable": sum(item.training == "train" for item in BUILTIN_FORMATS),
    }


def format_for_path(path: str | Path) -> FormatDescriptor | None:
    return FORMAT_BY_EXTENSION.get(Path(path).suffix.lower())


MODEL_FORMATS = tuple(item.to_dict() for item in BUILTIN_FORMATS if item.role == "model")
