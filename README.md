# scann-rk4-sj

RK4 小行星轨道传播器：**太阳引力 + 木星/土星点质量摄动 + 光行时迭代**，
输出 topocentric astrometric RA/Dec。从 SCANN 的 MPCORB 核验器中原样提取，
物理逻辑与实战版本一致，不依赖 SCANN 本体。

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

## 安装

```bash
pip install -e .
```

依赖：`numpy`、`astropy`。测试可选 `astroquery`（用于和 JPL Horizons 对比）。

## 命令行

### 全库核验（推荐，需要 MPCORB.DAT）

**本工具不下载、不附带 MPCORB 数据**。请自行到 Minor Planet Center 下载
全量轨道根数文件 [MPCORB.DAT](https://www.minorplanetcenter.net/iau/MPCORB.html)
（约 200 MB），把路径传给 `--mpcorb` 即可：

```bash
python -m scann_rk4_sj.cli verify \
  --mpcorb MPCORB.DAT \
  --target "03 07 29.679 +22 38 08.90" \
  --time 2023-10-15T10:07:16.52 \
  --radius 30
```

工具会自动解析并缓存（首次解析 1.2M 行需要一点时间，之后秒级加载），
内部走 SCANN 同款流程：两体粗筛 → RK4+木星/土星精细 → 光行时 → 命中判定。
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

批量预测用 `predict_refined`（所有参数支持数组），返回
`(ra, dec, mag, r_helio, delta)`。

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
- 与 JPL Horizons 对比（Ceres，网络可用时自动执行，容差 0.05°）。

## 来源与致谢

从 SCANN（Supernova Candidate Analysis via Neural Network）的
`MPCORBLocalVerifier` 提取，保持原实现数值逻辑不变。
