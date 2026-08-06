# scann-rk4-sj

小行星轨道传播与全库核验工具：**太阳引力 + 木星/土星点质量摄动 +
光行时迭代**，输出 topocentric astrometric RA/Dec，并支持用
MPCORB.DAT 全库对目标做命中判定。

## 功能

- 单颗预测：轨道根数 → 观测时刻的 RA/Dec/星等；
- 全库核验：用户提供 MPCORB.DAT + 目标坐标/时间 → HIT / MISS；
- MPCORB 80 列解析、坐标解析、命令行工具。

## 物理模型

1. 轨道根数（a, e, i, Ω, ω, M）→ 太阳中心黄道 J2000 状态向量（开普勒解）；
2. 旋转到赤道 J2000，RK4 固定步长积分：

   a = -GM_sun · r / r³ + Σ GM_p · ( (r_p − r)/|r_p − r|³ − r_p/|r_p|³ )

   摄动体：木星、土星；
3. 行星/太阳/地球位置来自 astropy builtin 星历（`solar_system_ephemeris='builtin'`）；
4. 观测者位置由 `EarthLocation` 转为日心赤道向量；
5. 光行时迭代（默认 4 次，容差 1e-8 天）得到 astrometric 方向。

注意：这是 topocentric **astrometric** 方向，未加光行差/章动，不是完整视位置；
精度适合小行星核验（角分到角秒量级），不适合做毫角秒级历表。

## 数据说明

**本工具不下载、不附带 MPCORB 数据**。请自行到 Minor Planet Center 下载
全量轨道根数文件 [MPCORB.DAT](https://www.minorplanetcenter.net/iau/MPCORB.html)
（约 200 MB）。行星/地球位置由 astropy builtin 星历计算，同样无需额外下载。

## 安装

```bash
pip install -e .
```

依赖：`numpy`、`astropy`。测试可选 `astroquery`（用于和 JPL Horizons 对比）。

## 命令行

### 全库核验（推荐）

```bash
python -m scann_rk4_sj.cli verify \
  --mpcorb MPCORB.DAT \
  --target "03 07 29.679 +22 38 08.90" \
  --time 2023-10-15T10:07:16.52 \
  --radius 30
```

流程：解析全库 → 两体粗筛（shortlist ≤400）→ RK4+木星/土星精细（≤300）
→ 光行时 → 命中判定。首次解析 1.2M 行需要一点时间，之后自动缓存、秒级加载。
输出 HIT 的小行星编号/名字、预测位置和角距离。

### 单颗预测

用轨道根数：

```bash
python -m scann_rk4_sj.cli predict \
  --elements "2.77,0.08,10.6,80.3,73.6,20.0" \
  --epoch 2026-01-01 --time 2026-08-06T18:00:00 \
  --H 3.34 --G 0.12 --site 87.179,43.471,2066
```

直接给 MPCORB.DAT 的一行：

```bash
python -m scann_rk4_sj.cli predict --mpcorb-line "00001 ..." --time 2026-08-06T18:00:00 --json
```

不传 `--site` 时默认用星明天文台 N89（东经 87.179°, 北纬 43.471°, 海拔 2066 m）。

### 命中判定（match）

拿 MPC 80 列 + 目标坐标，直接判断是否命中（默认 30″ 半径）：

```bash
python -m scann_rk4_sj.cli match \
  --mpcorb-line "00001 ..." \
  --time 2026-08-06T18:00:00 \
  --target "13:55:22.95,-06:11:30.2" \
  --radius 30
```

输出预测 RA/Dec、目标 RA/Dec、角距离和 HIT / MISS。目标坐标支持十进制度
（`208.845615,-6.191733`）或时分秒（`13:55:22.95,-06:11:30.2`）。

## Python API

```python
from scann_rk4_sj import AsteroidPropagator

res = AsteroidPropagator.predict_single(
    a=2.77, e=0.08, inc_deg=10.6, Omega_deg=80.3, w_deg=73.6, M_deg=20.0,
    epoch_iso="2026-01-01",
    obs_iso="2026-08-06T18:00:00",
    H=3.34, G=0.12,
    site_lon_deg=87.179, site_lat_deg=43.471, site_alt_m=2066,
)
print(res["ra_deg"], res["dec_deg"], res["mag"])
```

全库核验：

```python
from scann_rk4_sj import MpcorbVerifier

verifier = MpcorbVerifier("MPCORB.DAT")
results = verifier.verify_targets([
    {"ra_deg": 46.87366, "dec_deg": 22.63581,
     "time": "2023-10-15T10:07:16.52"},
], search_radius_arcsec=30)
print(results[0]["local_asteroid_status"])  # match / clear
```

MPCORB 行解析：

```python
from scann_rk4_sj import parse_mpcorb_line

els = parse_mpcorb_line(line)
# els 里直接有 a/e/inc/Omega/w/M0/H/G/epoch_iso
```

## 测试

```bash
python -m unittest discover -s tests -v
```

测试包括：

- 二体能量漂移（365 天 RK4 能量守恒）；
- 前向/后向往返传播自洽；
- 木星/土星摄动确实改变轨道（与纯二体对比）；
- 光行时迭代收敛；
- 与 JPL Horizons 对比（Ceres，网络可用时自动执行）；
- 全库核验的 HIT / clear 判定（合成小样本 MPCORB）。

## 致谢

感谢 **Minor Planet Center（MPC）** 提供并维护 **MPCORB.DAT** 全量小行星轨道
数据库；感谢 **JPL Horizons** 提供独立星历用于验证；感谢 astropy 项目提供
天文计算基础设施。
