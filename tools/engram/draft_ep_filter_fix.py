# DECIDER_V4_EXPERTMAP fix: the draft expert MAP is correct. What is wrong is the
# loader's pre-read EP weight filter, which is sized from the TARGET expert count.
#
#   filter window (model_config.get_num_experts() = n_routed_experts = 384, ep 4):
#       rank r keeps mtp.*.experts.E.*.weight only for E in [96r, 96r+96)
#   draft map window (dspark_n_routed_experts = 128, ep 4):
#       rank r owns E in [32r, 32r+32)
#   intersection: rank0 = 0..31 (all 32), rank1/2/3 = EMPTY
#
# So ranks 1-3 never receive their own draft expert .weight tensors (rank 1 receives
# 96..127, which its map correctly rejects = the "success=False" lines). Scales are not
# ".weight" names, the filter does not touch them, which is why scales loaded everywhere.
#
# Usage (inside the container, every node):
#   python3 epfilter_fix.py --probe-only   # zero-boot: print the filter source + window table
#   python3 epfilter_fix.py                # install
#   python3 epfilter_fix.py --undo         # restore *.pre-epfilter backups
#
# Installs:
#   1. ep_weight_filter.py: wrapper so mtp.* names are never pre-filtered (the draft's own
#      expert map drops non-local ids in RoutedExperts.weight_loader, as designed).
#      Live A/B without env (Ray worker env drifts): touch /tmp/epfilter.off = old behaviour.
#   2. dspark.py: _EPGATE per-param accounting (calls / ok / live slots) + fail-closed gate.
#      Opt-out: VLLM_DSPARK_ALLOW_UNLOADED_EXPERTS=1 or touch /tmp/epgate.allow
import ast
import os
import re
import shutil
import sys

ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"
F_FILTER = ROOT + "/model_executor/model_loader/ep_weight_filter.py"
F_LOADER = ROOT + "/model_executor/model_loader/default_loader.py"
F_DSPARK = ROOT + "/models/deepseek_v4_1/nvidia/dspark.py"
BAK = ".pre-epfilter"


def probe():
    print("=" * 30, F_FILTER)
    try:
        print(open(F_FILTER).read())
    except Exception as e:
        print("unreadable:", repr(e))
    print("=" * 30, F_LOADER, "(filter-sizing lines)")
    try:
        lines = open(F_LOADER).read().split("\n")
        pat = re.compile(
            r"local_expert_ids|get_num_experts|ep_weight_filter|compute_local_expert_ids"
        )
        hit = set()
        for i, ln in enumerate(lines):
            if pat.search(ln):
                hit.update(range(max(0, i - 2), min(len(lines), i + 3)))
        for i in sorted(hit):
            print("%5d  %s" % (i + 1, lines[i]))
    except Exception as e:
        print("unreadable:", repr(e))
    print("=" * 30, "window table (linear placement, ep_size 4)")
    for r in range(4):
        flt = set(range(96 * r, 96 * r + 96))
        own = set(range(32 * r, 32 * r + 32))
        arrive = sorted(flt & set(range(128)))
        both = sorted(flt & own)
        print(
            "rank %d: filter passes mtp experts %s (n=%d) | map owns %d..%d | loadable=%d of 32"
            % (
                r,
                ("%d..%d" % (arrive[0], arrive[-1])) if arrive else "none",
                len(arrive),
                32 * r,
                32 * r + 31,
                len(both),
            )
        )
    print("PREDICTION: only EP rank 0 has live draft experts; rank 1's first expert .weight")
    print("            to arrive is experts.100 (lexicographic first of 96..127).")


def undo():
    for f in (F_FILTER, F_DSPARK):
        if os.path.exists(f + BAK):
            shutil.copy2(f + BAK, f)
            print("restored", f)
        else:
            print("no backup for", f)


WRAP = '''

# _EPFILTER (DECIDER_V4_EXPERTMAP): this pre-read filter is sized from
# model_config.get_num_experts() (backbone n_routed_experts = 384), but the DSpark
# draft MoE under mtp.* is built with dspark_n_routed_experts (128). The 384 window
# [96r, 96r+96) only intersects the draft's own [32r, 32r+32) on EP rank 0, so ranks
# 1..3 never received their draft expert weights. Never pre-filter mtp.*: the draft's
# RoutedExperts.weight_loader already drops non-local ids through its (correct) map.
_epfilter_orig_should_skip_weight = should_skip_weight
_epfilter_state = {"exempt": 0}


def should_skip_weight(weight_name, local_expert_ids, *args, **kwargs):
    _skip = _epfilter_orig_should_skip_weight(
        weight_name, local_expert_ids, *args, **kwargs
    )
    if (
        _skip
        and isinstance(weight_name, str)
        and weight_name.startswith("mtp.")
        and not __import__("os").path.exists("/tmp/epfilter.off")
    ):
        _epfilter_state["exempt"] += 1
        if _epfilter_state["exempt"] == 1:
            print(
                "EPFILTER exempting mtp.* from the %d-id EP pre-filter (first=%s)"
                % (len(local_expert_ids), weight_name),
                flush=True,
            )
        return False
    return _skip
'''

GATE = '''        # _EPGATE: per-rank proof that every local draft expert slot was written.
        try:
            import os as _eo

            _ebad = []
            for _en, _ep in params_dict.items():
                if ".routed_experts." not in _en:
                    continue
                _ecalls, _eok = _EPG.get(_en, [0, 0])
                _eq = _ep.detach()
                if _eq.element_size() == 1 and _eq.dtype not in (torch.int8, torch.uint8):
                    _eq = _eq.view(torch.uint8)
                _eslots = int(_eq.shape[0])
                _elive = 0
                for _es in range(_eslots):
                    if bool(_eq[_es].ne(0).any()):
                        _elive += 1
                _eexp = _eslots * (2 if ".w13_" in _en else 1)
                logger.info(
                    "EPGATE rank=%d %s calls=%d ok=%d expect_ok=%d live_slots=%d/%d",
                    tp_rank, _en, _ecalls, _eok, _eexp, _elive, _eslots,
                )
                if _en.endswith(("_weight", "_weight_scale", "_weight_scale_inv")) and (
                    _elive != _eslots
                ):
                    _ebad.append("%s live=%d/%d ok=%d/%d" % (_en, _elive, _eslots, _eok, _eexp))
            try:
                _emgr = self.model.layers[0].ffn.experts.routed_experts.expert_map_manager
                _emap = getattr(_emgr, "expert_map", None)
                if _emap is not None:
                    _eown = (_emap >= 0).nonzero().flatten().tolist()
                    logger.info(
                        "EPGATE rank=%d map owns %d ids [%s..%s] map(first)=%s map(last)=%s",
                        tp_rank, len(_eown), _eown[0], _eown[-1],
                        _emgr.map_global_to_local(int(_eown[0])),
                        _emgr.map_global_to_local(int(_eown[-1])),
                    )
            except Exception as _ee:
                logger.info("EPGATE map probe failed: %r", _ee)
            if _ebad:
                _emsg = "EPGATE rank=%d FAIL dead draft expert slots: %s" % (tp_rank, _ebad)
                if (
                    _eo.environ.get("VLLM_DSPARK_ALLOW_UNLOADED_EXPERTS") == "1"
                    or _eo.path.exists("/tmp/epgate.allow")
                ):
                    logger.error(_emsg)
                else:
                    raise RuntimeError(_emsg)
            else:
                logger.info("EPGATE rank=%d PASS all draft expert slots live", tp_rank)
        except RuntimeError:
            raise
        except Exception as _ee:
            logger.info("EPGATE failed: %r", _ee)
'''


def install_filter():
    s = open(F_FILTER).read()
    if "_EPFILTER" in s:
        print("filter: already")
        return
    assert re.search(r"^def should_skip_weight\(", s, re.M), "should_skip_weight not found"
    shutil.copy2(F_FILTER, F_FILTER + BAK)
    s = s.rstrip("\n") + "\n" + WRAP
    ast.parse(s)
    open(F_FILTER, "w").write(s)
    print("VERIFIED ep_weight_filter.py _EPFILTER wrapper installed")


def install_gate():
    s = open(F_DSPARK).read()
    if "_EPGATE" in s:
        print("gate: already")
        return
    a1 = "        params_dict = dict(self.named_parameters())"
    a2 = (
        "                    if success:\n"
        "                        loaded_params.add(name_mapped)"
    )
    a3 = "        if self.model.confidence_head is not None and not loaded_confidence_head:"
    for tag, a in (("params_dict", a1), ("success", a2), ("tail", a3)):
        assert s.count(a) == 1, "anchor %s count=%d" % (tag, s.count(a))
    shutil.copy2(F_DSPARK, F_DSPARK + BAK)
    s = s.replace(a1, "        _EPG: dict = {}  # _EPGATE calls/ok per mapped param\n" + a1, 1)
    b2 = (
        "                    _epg = _EPG.setdefault(name_mapped, [0, 0])  # _EPGATE\n"
        "                    _epg[0] += 1\n"
        "                    if success:\n"
        "                        _epg[1] += 1\n"
        "                        loaded_params.add(name_mapped)"
    )
    s = s.replace(a2, b2, 1)
    s = s.replace(a3, GATE + a3, 1)
    ast.parse(s)
    open(F_DSPARK, "w").write(s)
    print("VERIFIED dspark.py _EPGATE accounting + fail-closed gate installed")


if __name__ == "__main__":
    if "--probe-only" in sys.argv:
        probe()
    elif "--undo" in sys.argv:
        undo()
    else:
        install_filter()
        install_gate()
