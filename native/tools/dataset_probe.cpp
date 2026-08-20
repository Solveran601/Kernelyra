// Safe, dependency-free format router for Kernelyra ingestion.
// It never executes or imports the target file as code. Decoders belong in
// separately sandboxed adapters (Tika, FFmpeg, libarchive, image codecs).
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Format { std::string name; std::string modality; };

std::string lower(std::string value) {
  for (auto& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

void add(std::unordered_map<std::string, Format>& out, const std::string& modality,
         const std::string& name, const std::string& extensions) {
  size_t start = 0;
  while (start < extensions.size()) {
    const auto end = extensions.find(' ', start);
    out.emplace(extensions.substr(start, end - start), Format{name, modality});
    start = end == std::string::npos ? extensions.size() : end + 1;
  }
}

std::unordered_map<std::string, Format> registry() {
  std::unordered_map<std::string, Format> formats;
  add(formats, "table", "delimited", "csv tsv tab psv ssv dat");
  add(formats, "table", "json", "json jsonl ndjson geojson topojson");
  add(formats, "table", "columnar", "parquet arrow feather orc avro arff libsvm svm");
  add(formats, "table", "spreadsheet", "xls xlsx xlsm xlsb ods numbers wk1 wk2 wk3 wks dbf dif slk");
  add(formats, "text", "plain-text", "txt text log rst adoc org tex rtf");
  add(formats, "text", "markup", "md markdown html htm xhtml xml yaml yml toml ini cfg conf properties");
  add(formats, "text", "source-code", "py pyw js mjs cjs ts tsx jsx java kt kts c cc cpp cxx h hpp cs go rs rb php swift scala sh bash zsh ps1 sql r lua dart pl pm vb fs fsx clj hs ex exs");
  add(formats, "document", "office", "doc docx dot dotx odt ott ppt pptx pps ppsx pot potx odp otp pages key epub mobi azw azw3 fb2");
  add(formats, "document", "pdf", "pdf xps oxps djvu ps eps");
  add(formats, "image", "raster", "png jpg jpeg jpe jfif gif bmp dib tif tiff webp heic heif avif ico cur jp2 j2k jpf jpx ppm pgm pbm pnm tga dds hdr exr raw cr2 nef arw dng raf orf rw2");
  add(formats, "image", "vector", "svg svgz ai cdr eps ps emf wmf");
  add(formats, "audio", "audio", "wav wave mp3 flac ogg oga opus aac m4a wma aiff aif au amr ape alac mid midi kar");
  add(formats, "video", "video", "mp4 m4v mov mkv webm avi mpg mpeg m2v ts m2ts mts vob flv f4v wmv asf 3gp 3g2 ogv rm ram");
  add(formats, "archive", "archive", "zip zipx tar tgz gz gzip bz2 bzip2 xz lz lzma zst 7z rar cab arj lzh lha cpio iso dmg pkg deb rpm apk jar war ear");
  add(formats, "database", "database", "sqlite sqlite3 db mdb accdb fdb gdb ibd myd frm bak sqlitedb");
  add(formats, "scientific", "numeric-scientific", "mat hdf hdf4 hdf5 nc netcdf cdf fits fit fts sav sas7bdat xpt por dta rdata rds fst npy npz mtx h5ad loom zarr pickle pkl joblib");
  add(formats, "geospatial", "geo-cad-point-cloud", "shp shx prj qpj kml kmz gpx gml gpkg asc dem las laz e57 ply pcd obj stl gltf glb dae fbx 3ds blend step stp iges igs dwg dxf");
  add(formats, "bio-medical", "bio-medical", "fasta fa fna faa fastq fq bam sam cram vcf bcf gff gff3 gtf bed bigwig bw bigbed bb mzml mzxml dcm dicom nii nrrd mhd mha");
  add(formats, "structured", "binary-structured", "msgpack mpk bson cbor ubjson protobuf proto pb capnp thrift flatbuffers fbs");
  add(formats, "binary", "executable", "exe dll sys msi com bat cmd ps1 app elf so dylib bin");
  return formats;
}

std::string escape(const std::string& value) {
  std::string out;
  for (char c : value) {
    if (c == '\\' || c == '"') out += '\\';
    if (c == '\n') out += "\\n"; else if (c != '\r') out += c;
  }
  return out;
}

bool starts(const std::vector<unsigned char>& data, std::initializer_list<unsigned char> sig) {
  if (data.size() < sig.size()) return false;
  size_t i = 0; for (auto value : sig) if (data[i++] != value) return false;
  return true;
}

Format sniff(const fs::path& path, const std::unordered_map<std::string, Format>& formats) {
  std::ifstream input(path, std::ios::binary);
  std::vector<unsigned char> bytes(16);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  bytes.resize(static_cast<size_t>(input.gcount()));
  if (starts(bytes, {0x89,'P','N','G'})) return {"png", "image"};
  if (starts(bytes, {0xff,0xd8,0xff})) return {"jpeg", "image"};
  if (starts(bytes, {'G','I','F','8'})) return {"gif", "image"};
  if (starts(bytes, {'%','P','D','F','-'})) return {"pdf", "document"};
  if (starts(bytes, {'P','K',3,4})) return {"zip-container", "archive"};
  if (starts(bytes, {'P','A','R','1'})) return {"parquet", "table"};
  if (starts(bytes, {'S','Q','L','i','t','e'})) return {"sqlite", "database"};
  if (starts(bytes, {'R','I','F','F'})) return {"riff", "audio"};
  if (starts(bytes, {'I','D','3'})) return {"mp3", "audio"};
  if (starts(bytes, {0x7f,'E','L','F'})) return {"elf", "binary"};
  if (starts(bytes, {'M','Z'})) return {"pe", "binary"};
  const auto extension = lower(path.extension().string());
  if (const auto found = formats.find(extension.size() > 1 ? extension.substr(1) : ""); found != formats.end()) return found->second;
  return {"unknown", "binary"};
}

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: dataset_probe [--json] <file> [file...]\n"; return 2; }
  const bool single_json = std::string(argv[1]) == "--json";
  const int first_path = single_json ? 2 : 1;
  if (argc <= first_path) { std::cerr << "--json requires exactly one file\n"; return 2; }
  if (single_json && argc != first_path + 1) { std::cerr << "--json accepts exactly one file\n"; return 2; }
  const auto formats = registry();
  if (!single_json) std::cout << "[";
  for (int i = first_path; i < argc; ++i) {
    const fs::path path = argv[i];
    std::error_code error;
    const bool regular = fs::is_regular_file(path, error);
    const auto format = regular ? sniff(path, formats) : Format{"unreadable", "binary"};
    if (!single_json && i > first_path) std::cout << ',';
    std::cout << "{\"protocol\":\"kernelyra-native-probe/1\",\"path\":\"" << escape(path.string()) << "\",\"bytes\":"
              << (regular ? fs::file_size(path, error) : 0) << ",\"format\":\"" << format.name
              << "\",\"modality\":\"" << format.modality
              << "\",\"engine\":\"native-cpp\",\"safe\":true}";
  }
  if (!single_json) std::cout << "]";
  std::cout << "\n";
}
