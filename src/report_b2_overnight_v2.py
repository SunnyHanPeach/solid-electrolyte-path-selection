"""Write the B2 report from verified artifacts, with no model execution."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import median

from audit_signed_barriers import rows

BEIJING = timezone(timedelta(hours=8))
CLASSIFICATION = {
    "positive": "通过正收益门槛", "negative": "通过退化门槛",
    "near_zero": "实际近零", "unresolved": "未决",
}


def duration(seconds):
    minutes = round(seconds/60)
    return f"{minutes//60} 小时 {minutes%60} 分钟" if minutes >= 60 else f"{minutes} 分钟"


def window(value):
    return datetime.fromisoformat(value.replace("Z","+00:00")).astimezone(BEIJING).strftime("%m-%d %H:%M")


def effect(row):
    return (f"{float(row['mean_meV']):+.2f} [{float(row['low95_meV']):+.2f}, "
            f"{float(row['high95_meV']):+.2f}]")


def report(folder, plan_file):
    plan=json.loads(plan_file.read_bytes())
    status=json.loads((folder/"status.json").read_bytes())
    verification=json.loads((folder/"verification.json").read_bytes())
    if status["state"]!="complete" or verification["status"]!="pass":
        raise ValueError("Final report requires verified run evidence")
    interval=rows((folder/"intervals.csv").read_bytes())
    diagnostics=rows((folder/"training_diagnostics.csv").read_bytes())
    by_path={(int(r["slot"]),r["metric"]):r for r in interval if r["analysis"]=="regular"}
    evaluated=status["completed_paths"]
    both=[plan["candidates"][i-1]["material_id"] for i in evaluated if all(
          by_path[i,m]["classification"]=="positive" for m in ("absolute","net"))]
    bins={m:Counter(by_path[i,m]["classification"] for i in evaluated) for m in ("absolute","net")}
    running_s=sum(float(r["fit_seconds"]) for r in diagnostics)
    median_s=median(float(r["fit_seconds"]) for r in diagnostics)
    scored=verification["evaluation_materials"]
    fitted=status["completed_fits"]
    lines=["# B2-Overnight-01 最终研究报告","",
           f"完成 **{len(evaluated)} 条路径、{fitted}/{plan['planned_fits']} 次训练**。"
           f"逐材料表在同一 {scored} 材料面板经独立复算。附加种子"
           f"{'已完成' if status['extra_replicated'] else '未纳入本批已完成统计'}。","",
           "## 路径训练收益","",
           "所有值均为 meV。正数表示相比参照降低了正向能垒 MAE。"
           "区间以路径内三个配对种子的均值为输入，按评估化学体系重采样。","",
           "| 路径 | 化学体系 | 绝对收益 95% 区间 | 净收益 95% 区间 | 判定（绝对／净） |",
           "|---|---|---:|---:|---|"]
    for i in evaluated:
        p=plan["candidates"][i-1]
        aa,nn=by_path[i,"absolute"],by_path[i,"net"]
        lines.append(f"| {p['material_id']} | {p['chemsys']} | {effect(aa)} | {effect(nn)} | "
                     f"{CLASSIFICATION[aa['classification']]}／{CLASSIFICATION[nn['classification']]} |")
    lines += ["",f"两项均通过的路径为 {', '.join(both)}。" if both else "本轮没有路径同时通过两项正收益门槛。"]
    for m,title in (("absolute","绝对收益"),("net","净收益")):
        counts="、".join(f"{CLASSIFICATION[c]} {bins[m][c]} 条" for c in CLASSIFICATION)
        lines.append(f"{title}判定为{counts}。")
    extra=[r for r in interval if r["analysis"]!="regular"]
    if extra:
        lines += ["","### mp-753143 额外种子复验","",
                   "| 种子 | 收益 | 均值与 95% 区间（meV） | 判定 |",
                   "|---|---|---:|---|"]
        for r in extra:
            lines.append(f"| {r['seeds'].replace(';','、')} | {'绝对' if r['metric']=='absolute' else '净'} | "
                         f"{effect(r)} | {CLASSIFICATION[r['classification']]} |")
    early=verification["legacy_component_unavailable"]
    lines += ["","## 运行时间和日志","",
              f"- 北京时间 {window(status['prior_start_utc'])} 开始，{window(status['ended_utc'])} 完成统一评分。"
              f"机器经过时间 {duration(verification['elapsed_seconds'])}，其中推理和评分 "
              f"{duration(verification['scoring_seconds'])}。单次训练中位 {duration(median_s)}，"
              f"所有任务的训练耗时之和为 {duration(running_s)}，包含同时运行的任务。",
              "- 实例实际提供一张 22 GiB RTX 2080 Ti。常规队列采用同卡双进程，排程和训练配方保持冻结。",
              "- 每次训练的梯度范数、裁剪频率与终点面板 MAE 均已归档。"]
    if early:
        lines[-1]+=f" 早期沿用的 {len(early)} 次已完任务仅有合并损失，能量和力分项留空。"
    if status["skipped"]:
        lines.append(f"- 因时间预留停发的区块记录在状态清单，共 {len(status['skipped'])} 项。")
    lines += ["","## 证据边界","",
              "本次使用同一历史留出面板，指标反映冻结配方下的路径训练价值。"
              "纯 D₀ 权重源于历史 RTX 4090 训练，并随本批模型在同一运行配置下重推理。"
              "净收益的 Path 与 Replay 均来自当前实例。区间描述材料组成的变化，逐种子收益另列。"
              "84 次预计划训练覆盖多个候选，单路径区间未做多重比较校正。"
              "当前实验积累路径价值标签，DFT 前选样效果和同实际核时的优势仍需独立验证。"
              "没有启动后续实验。","",
              "[绝对与净收益](intervals.csv) · [逐材料收益](material_gains.csv) · "
              "[训练诊断与计时](training_diagnostics.csv) · [检查点哈希](checkpoint_hashes.csv) · "
              "[独立复算](verification.json)"]
    with (folder/"final_report.md").open("x",encoding="utf-8",newline="\n") as f:
        f.write("\n".join(lines)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("plan_file",type=Path)
    report(**vars(p.parse_args()))
