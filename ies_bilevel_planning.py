#!/usr/bin/env python3
"""双层规划：氢-氨-醇一体化零净排放综合能源系统（IES）.

上层：容量配置（通过枚举+下层优化近似求解）
下层：给定容量后的逐时调度（Gurobi LP）

运行示例：
    python ies_bilevel_planning.py --hours 24 --sensitivity --compare
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 gurobipy，请先安装 Gurobi Python API 并配置许可证。"
    ) from exc


@dataclass
class TechCost:
    capex: float
    fixed_om: float


@dataclass
class PlanningConfig:
    discount_rate: float = 0.08
    years: int = 20
    co2_factor_grid: float = 0.58  # tCO2/MWh
    carbon_price: float = 350.0  # CNY/tCO2
    electricity_price: float = 520.0  # CNY/MWh
    gas_price: float = 230.0  # CNY/MWh-LHV
    capture_cost: float = 120.0  # CNY/tCO2 captured

    # 设备投资成本（单位统一按 MW 或 MWh 容量）
    costs: Dict[str, TechCost] = None

    def __post_init__(self):
        if self.costs is None:
            self.costs = {
                "wind": TechCost(capex=4.8e6, fixed_om=0.02),
                "pv": TechCost(capex=3.6e6, fixed_om=0.015),
                "electrolyzer": TechCost(capex=2.2e6, fixed_om=0.025),
                "h2_storage": TechCost(capex=1.2e6, fixed_om=0.02),
                "ammonia": TechCost(capex=2.8e6, fixed_om=0.03),
                "methanol": TechCost(capex=3.2e6, fixed_om=0.03),
                "ccus": TechCost(capex=2.5e6, fixed_om=0.03),
                "battery": TechCost(capex=1.0e6, fixed_om=0.02),
            }


@dataclass
class TimeSeries:
    load_e: List[float]
    load_h2: List[float]
    load_nh3: List[float]
    load_meoh: List[float]
    wind_cf: List[float]
    pv_cf: List[float]


def annuity_factor(rate: float, years: int) -> float:
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def build_profiles(hours: int) -> TimeSeries:
    load_e, load_h2, load_nh3, load_meoh, wind_cf, pv_cf = [], [], [], [], [], []
    for t in range(hours):
        h = t % 24
        daily_wave = 0.5 + 0.5 * math.sin((h - 8) / 24 * 2 * math.pi)
        load_e.append(70 + 20 * daily_wave)
        load_h2.append(8 + 2 * (1 - daily_wave))
        load_nh3.append(5 + 1.5 * daily_wave)
        load_meoh.append(4 + 1.0 * daily_wave)

        wind_cf.append(max(0.15, min(0.85, 0.45 + 0.2 * math.sin((h + 4) / 24 * 2 * math.pi))))
        pv_cf.append(max(0.0, math.sin((h - 6) / 12 * math.pi)))
    return TimeSeries(load_e, load_h2, load_nh3, load_meoh, wind_cf, pv_cf)


def solve_lower_dispatch(cap: Dict[str, float], ts: TimeSeries, cfg: PlanningConfig) -> Dict[str, float]:
    T = range(len(ts.load_e))
    m = gp.Model("lower_dispatch")
    m.Params.OutputFlag = 0

    p_grid = m.addVars(T, lb=0, name="p_grid")
    p_wind = m.addVars(T, lb=0, name="p_wind")
    p_pv = m.addVars(T, lb=0, name="p_pv")
    p_elz = m.addVars(T, lb=0, name="p_elz")

    b_ch = m.addVars(T, lb=0, name="b_ch")
    b_dis = m.addVars(T, lb=0, name="b_dis")
    soc = m.addVars(T, lb=0, name="soc")

    h2_prod = m.addVars(T, lb=0, name="h2_prod")
    h2_to_nh3 = m.addVars(T, lb=0, name="h2_to_nh3")
    h2_to_meoh = m.addVars(T, lb=0, name="h2_to_meoh")
    h2_direct = m.addVars(T, lb=0, name="h2_direct")
    h2_ch = m.addVars(T, lb=0, name="h2_ch")
    h2_dis = m.addVars(T, lb=0, name="h2_dis")
    h2_inv = m.addVars(T, lb=0, name="h2_inv")

    nh3_prod = m.addVars(T, lb=0, name="nh3_prod")
    meoh_prod = m.addVars(T, lb=0, name="meoh_prod")

    gas_use = m.addVars(T, lb=0, name="gas_use")
    co2_emit = m.addVars(T, lb=0, name="co2_emit")
    co2_captured = m.addVars(T, lb=0, name="co2_captured")

    eta_elz = 0.68  # MWh_H2 / MWh_e
    eta_batt = 0.92
    eta_nh3 = 0.78
    eta_meoh = 0.74
    co2_gas_factor = 0.22  # tCO2/MWh gas

    for t in T:
        m.addConstr(p_wind[t] <= cap["wind"] * ts.wind_cf[t])
        m.addConstr(p_pv[t] <= cap["pv"] * ts.pv_cf[t])
        m.addConstr(p_elz[t] <= cap["electrolyzer"])

        m.addConstr(h2_prod[t] == eta_elz * p_elz[t])
        m.addConstr(nh3_prod[t] == eta_nh3 * h2_to_nh3[t])
        m.addConstr(meoh_prod[t] == eta_meoh * h2_to_meoh[t])

        m.addConstr(nh3_prod[t] >= ts.load_nh3[t])
        m.addConstr(meoh_prod[t] >= ts.load_meoh[t])
        m.addConstr(h2_direct[t] >= ts.load_h2[t])

        m.addConstr(h2_prod[t] + h2_dis[t] == h2_to_nh3[t] + h2_to_meoh[t] + h2_direct[t] + h2_ch[t])

        m.addConstr(p_grid[t] + p_wind[t] + p_pv[t] + b_dis[t] == ts.load_e[t] + p_elz[t] + b_ch[t])

        m.addConstr(gas_use[t] == 0.16 * (nh3_prod[t] + meoh_prod[t]))
        m.addConstr(co2_emit[t] == cfg.co2_factor_grid * p_grid[t] + co2_gas_factor * gas_use[t])
        m.addConstr(co2_captured[t] <= co2_emit[t])
        m.addConstr(co2_captured[t] <= cap["ccus"])

        m.addConstr(b_ch[t] <= cap["battery"])
        m.addConstr(b_dis[t] <= cap["battery"])

        m.addConstr(h2_ch[t] <= cap["h2_storage"])
        m.addConstr(h2_dis[t] <= cap["h2_storage"])
        m.addConstr(h2_inv[t] <= cap["h2_storage"])

    for t in T:
        prev = len(T) - 1 if t == 0 else t - 1
        m.addConstr(soc[t] == soc[prev] + eta_batt * b_ch[t] - b_dis[t] / eta_batt)
        m.addConstr(soc[t] <= cap["battery"])
        m.addConstr(h2_inv[t] == h2_inv[prev] + h2_ch[t] - h2_dis[t])

    op_cost = gp.quicksum(
        cfg.electricity_price * p_grid[t]
        + cfg.gas_price * gas_use[t]
        + cfg.capture_cost * co2_captured[t]
        + cfg.carbon_price * (co2_emit[t] - co2_captured[t])
        for t in T
    )
    m.setObjective(op_cost, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return {"feasible": 0, "op_cost": float("inf")}

    total_emit = sum(co2_emit[t].X for t in T)
    total_capture = sum(co2_captured[t].X for t in T)

    return {
        "feasible": 1,
        "op_cost": m.ObjVal,
        "emission": total_emit,
        "capture": total_capture,
        "net_emission": total_emit - total_capture,
    }


def upper_capacity_search(ts: TimeSeries, cfg: PlanningConfig, net_zero_eps: float = 1e-3):
    af = annuity_factor(cfg.discount_rate, cfg.years)

    grid = {
        "wind": [30, 45, 60],
        "pv": [30, 45, 60],
        "electrolyzer": [14, 20, 26],
        "h2_storage": [14, 22, 30],
        "ammonia": [8, 11, 14],
        "methanol": [7, 10, 13],
        "ccus": [0, 10, 20],
        "battery": [10, 20, 30],
    }

    best = None
    keys = list(grid.keys())

    def recurse(i: int, cap: Dict[str, float]):
        nonlocal best
        if i == len(keys):
            lower = solve_lower_dispatch(cap, ts, cfg)
            if not lower["feasible"]:
                return

            if lower["net_emission"] > net_zero_eps:
                return

            invest = 0.0
            for k, v in cap.items():
                c = cfg.costs[k]
                invest += v * c.capex * (af + c.fixed_om)
            total = invest + lower["op_cost"]

            cand = {
                "capacity": dict(cap),
                "invest_cost": invest,
                "op_cost": lower["op_cost"],
                "total_cost": total,
                "net_emission": lower["net_emission"],
                "capture": lower["capture"],
            }
            if best is None or cand["total_cost"] < best["total_cost"]:
                best = cand
            return

        k = keys[i]
        for v in grid[k]:
            cap[k] = v
            recurse(i + 1, cap)

    recurse(0, {})
    return best


def run_comparison(ts: TimeSeries, cfg: PlanningConfig):
    baseline = PlanningConfig(**{**cfg.__dict__})
    best_zero = upper_capacity_search(ts, cfg)

    cfg_no_ccus = PlanningConfig(**{**cfg.__dict__})
    cfg_no_ccus.costs = dict(cfg.costs)

    # 约束 CCUS = 0
    def no_ccus_search():
        af = annuity_factor(cfg_no_ccus.discount_rate, cfg_no_ccus.years)
        best = None
        for wind in [20, 30, 40, 50, 60]:
            for pv in [20, 30, 40, 50, 60]:
                for elz in [12, 16, 20, 24, 28]:
                    cap = {
                        "wind": wind,
                        "pv": pv,
                        "electrolyzer": elz,
                        "h2_storage": 18,
                        "ammonia": 10,
                        "methanol": 9,
                        "ccus": 0,
                        "battery": 16,
                    }
                    lower = solve_lower_dispatch(cap, ts, cfg_no_ccus)
                    if not lower["feasible"]:
                        continue
                    invest = sum(cap[k] * cfg_no_ccus.costs[k].capex * (af + cfg_no_ccus.costs[k].fixed_om) for k in cap)
                    total = invest + lower["op_cost"]
                    cand = {"cap": cap, "total": total, "net": lower["net_emission"]}
                    if best is None or cand["total"] < best["total"]:
                        best = cand
        return best

    return {
        "zero_net_case": best_zero,
        "no_ccus_case": no_ccus_search(),
        "_note": baseline.discount_rate,
    }


def run_sensitivity(ts: TimeSeries, cfg: PlanningConfig):
    out = []
    for cp in [200, 300, 400, 500]:
        cfg_i = PlanningConfig(**{**cfg.__dict__})
        cfg_i.carbon_price = cp
        best = upper_capacity_search(ts, cfg_i)
        out.append((cp, best["total_cost"], best["capacity"]["ccus"], best["capacity"]["electrolyzer"]))
    return out


def pretty_print(best):
    print("\n===== 最优上层容量配置 =====")
    for k, v in best["capacity"].items():
        print(f"{k:12s}: {v:6.2f}")
    print("\n===== 经济性与排放 =====")
    print(f"年化投资成本: {best['invest_cost']:.2f}")
    print(f"运行调度成本: {best['op_cost']:.2f}")
    print(f"总成本      : {best['total_cost']:.2f}")
    print(f"净排放      : {best['net_emission']:.6f} tCO2")
    print(f"捕集总量    : {best['capture']:.4f} tCO2")


def main():
    parser = argparse.ArgumentParser(description="氢氨醇一体化IES双层优化规划（Gurobi）")
    parser.add_argument("--hours", type=int, default=24, help="优化时域长度（小时）")
    parser.add_argument("--compare", action="store_true", help="运行对比案例（无CCUS）")
    parser.add_argument("--sensitivity", action="store_true", help="运行碳价敏感性分析")
    args = parser.parse_args()

    cfg = PlanningConfig()
    ts = build_profiles(args.hours)
    best = upper_capacity_search(ts, cfg)
    if best is None:
        raise SystemExit("未找到满足净零排放约束的可行解，请扩大容量搜索空间。")

    pretty_print(best)

    if args.compare:
        comp = run_comparison(ts, cfg)
        print("\n===== 对比案例 =====")
        print(f"零净排放总成本: {comp['zero_net_case']['total_cost']:.2f}")
        print(f"无CCUS最优总成本: {comp['no_ccus_case']['total']:.2f}")
        print(f"无CCUS净排放: {comp['no_ccus_case']['net']:.4f} tCO2")

    if args.sensitivity:
        print("\n===== 碳价敏感性分析 =====")
        for cp, total, ccus_cap, elz_cap in run_sensitivity(ts, cfg):
            print(f"碳价={cp:>4.0f}: 总成本={total:>12.2f}, CCUS={ccus_cap:>5.1f}, ELZ={elz_cap:>5.1f}")


if __name__ == "__main__":
    main()
