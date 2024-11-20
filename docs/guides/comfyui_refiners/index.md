---
icon: material/vector-line
---

We provide a set of custom nodes for ComfyUI that allow you to easily use some of Refiners' models and utilities in your ComfyUI workflows.

## Installation

The nodes are published at <https://registry.comfy.org/publishers/finegrain/nodes/comfyui-refiners>.

### Requirements

1\. Install python (>=3.10):
=== "Linux"
    ```bash
    sudo apt install python3
    ```
=== "Windows"
    ```powershell
    winget install -e --id Python.Python.3.11
    ```

2\. Install git:
=== "Linux"
    ```bash
    sudo apt install git
    ```
=== "Windows"
    ```powershell
    winget install -e --id Git.Git
    ```

3\. (Optional) Create a python virtual environment:
=== "Linux"
    ```bash
    ~/Documents/comfy$ python -m venv .venv
    ~/Documents/comfy$ source .venv/bin/activate
    ```
=== "Windows"
    ```powershell
    PS C:\Users\Laurent\Documents\comfy> python -m venv .venv
    PS C:\Users\Laurent\Documents\comfy> .\.venv\Scripts\activate
    ```

4\. Install [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started):
=== "Linux"
    ```bash
    (.venv) ~/Documents/comfy$ pip install comfy-cli
    ```
=== "Windows"
    ```powershell
    (.venv) PS C:\Users\Laurent\Documents\comfy> pip install comfy-cli
    ```

5\. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI):
=== "Linux"
    ```bash
    (.venv) ~/Documents/comfy$ comfy --here install
    ```
=== "Windows"
    ```powershell
    (.venv) PS C:\Users\Laurent\Documents\comfy> comfy --here install
    ```

### Comfy Registry (recommended)

1\. Install the [comfyui-refiners](https://registry.comfy.org/publishers/finegrain/nodes/comfyui-refiners) custom nodes:
=== "Linux"
    ```bash
    (.venv) ~/Documents/comfy$ comfy node registry-install comfyui-refiners
    ```
=== "Windows"
    ```powershell
    (.venv) PS C:\Users\Laurent\Documents\comfy> comfy node registry-install comfyui-refiners
    ```

2\. Ensure that comfyui-refiners's dependencies are installed:
=== "Linux"
    ```bash
    (.venv) ~/Documents/comfy$ pip install -r ./ComfyUI/custom_nodes/comfyui-refiners/requirements.txt
    ```
=== "Windows"
    ```powershell
    (.venv) PS C:\Users\Laurent\Documents\comfy> pip install -r .\ComfyUI\custom_nodes\comfyui-refiners\requirements.txt
    ```

3\. Start ComfyUI:
=== "Linux"
    ```bash
    (.venv) ~/Documents/comfy$ comfy launch
    ```
=== "Windows"
    ```powershell
    (.venv) PS C:\Users\Laurent\Documents\comfy> comfy launch
    ```

### Manually

To manually install the nodes, you may alternatively do the following:

1. Download an archive of the nodes by cliking the "Download Latest" button at <https://registry.comfy.org/publishers/finegrain/nodes/comfyui-refiners>, or by running the following command:
```shell
curl -o comfyui-refiners.zip https://storage.googleapis.com/comfy-registry/finegrain/comfyui-refiners/1.0.4/node.tar.gz
```

2. Extract the archive to the `custom_nodes` directory:
```shell
unzip -d custom_nodes/comfyui-refiners comfyui-refiners.zip
```

3. Install the nodes' dependencies:
```shell
pip install -r custom_nodes/comfyui-refiners/requirements.txt
```

4. Remove the (now useless) downloaded archive:
```shell
rm comfyui-refiners.zip
```
