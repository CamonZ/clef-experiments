EMLX_PRIV = $(MIX_APP_PATH)/../emlx/priv
MLX_INCLUDE = $(EMLX_PRIV)/mlx/include
MLX_LIB = $(EMLX_PRIV)/mlx/lib
PLUGIN_INCLUDE = deps/emlx/emlx/c_src
TARGET = $(MIX_APP_PATH)/priv/libclef.so

all: $(TARGET)

$(TARGET): c_src/clef_plugin.cpp $(PLUGIN_INCLUDE)/emlx/plugin/abi.hpp $(MLX_INCLUDE)/mlx/fast.h
	mkdir -p "$(MIX_APP_PATH)/priv"
	$(CXX) -std=c++20 -O3 -fPIC -fvisibility=hidden -mmacosx-version-min=26.2 -isystem "$(MLX_INCLUDE)" -I"$(PLUGIN_INCLUDE)" c_src/clef_plugin.cpp -o "$@" -L"$(MLX_LIB)" -lmlx -shared -undefined dynamic_lookup -flat_namespace -rpath @loader_path/../../emlx/priv/mlx/lib -framework Metal -framework Foundation -framework Accelerate

.PHONY: all
