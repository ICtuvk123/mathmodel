# Cylindrical 运行指令

当前自定义 view 名称是 `cylindrical`。

`--view_args` 的格式是：

```bash
radius_ratio,theta_deg
```

默认值是：

```bash
0.2,180
```

建议先不要开 `--generate_1024`，先把 `64 -> 256` 的结果跑通。

## 推荐环境

推荐在 Linux + CUDA 下运行。

仓库原始环境文件：

```bash
cd visual_anagrams
conda env create -f environment.yml
conda activate visual_anagrams
python huggingface_login.py
```

如果你当前环境缺依赖，最少先补这个：

```bash
pip install einops
```

## 进入仓库

```bash
cd visual_anagrams
```

## 1. 纯 prompt 版本

纸面和镜面都用 prompt 约束：

```bash
python generate.py \
  --name cylindrical_prompt_prompt \
  --prompts "a lithograph of a cat" "a charcoal sketch of a woman's face" \
  --views identity cylindrical \
  --view_args 0.2,180 \
  --num_samples 1 \
  --num_inference_steps 30 \
  --guidance_scale 10.0
```

说明：

- 第一个 prompt 对应纸面 `identity`
- 第二个 prompt 对应镜面 `cylindrical`

## 2. 镜面图像引导 + 纸面 prompt

如果你手里已经有“希望在圆柱镜里看到的图”，优先跑这个。

```bash
python generate.py \
  --name cylindrical_ref_prompt \
  --prompts "" "a woodcut portrait on paper" \
  --views cylindrical identity \
  --view_args 0.2,180 none \
  --ref_im_path ../mona.png \
  --num_samples 1 \
  --num_inference_steps 30 \
  --guidance_scale 10.0
```

说明：

- `--views cylindrical identity` 的顺序不要反
- 因为当前仓库的 `ref_im_path` 机制会固定第一个 component
- 所以镜面目标图要挂在第一个 view，也就是 `cylindrical`
- 第二个 prompt 是纸面最终想看到的语义

如果你的纸面目标想更艺术化，可以改成：

```bash
--prompts "" "a surreal woodcut poster on paper"
```

## 3. 调参示例

更小镜面半径：

```bash
--view_args 0.15,180
```

更大可视角度：

```bash
--view_args 0.2,240
```

更窄可视角度：

```bash
--view_args 0.2,120
```

## 4. Windows 备注

Windows 下如果你只是想试运行，可以先这样：

```powershell
cd "C:\Users\12831\Desktop\C++ proj\mathmodel\visual_anagrams"
pip install einops
python generate.py --name cylindrical_prompt_prompt --prompts "a lithograph of a cat" "a charcoal sketch of a woman's face" --views identity cylindrical --view_args 0.2,180 --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0
```

但正式跑大模型时，还是建议 Linux。

## 5. 输出位置

结果会保存在：

```bash
visual_anagrams/results/<name>/
```

例如：

```bash
visual_anagrams/results/cylindrical_ref_prompt/
```

## 6. 当前已知限制

- `cylindrical` 是我接入到 Visual Anagrams 的第一版近似实现
- 这版已经能作为新 view 跑起来，但还没有做 LookingGlass 那种专门针对非正交 warp 的补偿
- `--generate_1024` 暂时不建议开，尤其是你用 `ref_im_path` 的时候
