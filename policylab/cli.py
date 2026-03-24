"""CLI entry point for the PolicyLab.

Usage:
    python -m policylab.cli stress-test --policy "AI Registration Act" --description "..."
    python -m policylab.cli war-game --incident "Deepfake Election Interference"
    python -m policylab.cli blind-spots --framework moderate --n-scenarios 50
    python -m policylab.cli backtest --case eu-ai-act
    python -m policylab.cli demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from concordia.language_model import language_model as lm_lib


def _get_model() -> lm_lib.LanguageModel:
    """Get LLM backend from env vars or fallback to no-op mock."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    model_name = os.environ.get("POLICYLAB_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from policylab.llm_backend import OpenAIModel
            label = f"{model_name}"
            if base_url:
                label += f" @ {base_url}"
            print(f"[LLM] {label}")
            return OpenAIModel(
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
        except Exception as e:
            print(f"[LLM] Failed: {e}")

    from concordia.language_model import no_language_model
    print("[LLM] No API key — using mock model (set OPENAI_API_KEY for real LLM)")
    return no_language_model.NoLanguageModel()


def _get_embedder():
    """Load sentence-transformers embedder or fallback to random hash."""
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Embedder] Using all-MiniLM-L6-v2")
        return lambda text: _model.encode(text)
    except ImportError:
        print("[Embedder] Using random embedder")
        def embedder(text: str) -> np.ndarray:
            rng = np.random.RandomState(hash(text) % 2**31)
            return rng.randn(384).astype(np.float32)
        return embedder


def cmd_stress_test(args):
    """Run stress test with SAFETY_FIRST_CORP contrarian and ensemble perturbation."""
    from policylab.features.stress_tester import StressTester

    model = _get_model()
    embedder = _get_embedder()

    tester = StressTester(
        model=model,
        embedder=embedder,
        n_ensemble=args.n_ensemble,
        num_rounds=args.rounds,
        output_dir=args.output,
        revalidate=getattr(args, "revalidate", False),
    )

    if getattr(args, "revalidate", False):
        print("[Revalidation] ON — each action will be confirmed by LLM before resolution")

    report = tester.stress_test(
        policy_name=args.policy,
        policy_description=args.description,
        regulated_entities=args.entities.split(",") if args.entities else ["AI companies"],
        requirements=args.requirements.split(";") if args.requirements else ["Comply with regulation"],
        penalties=args.penalties.split(";") if args.penalties else ["Fines up to $10M"],
    )

    report.print_summary()


def cmd_war_game(args):
    """Run war-game across the three governance framework presets."""
    from policylab.features.war_game import (
        WarGame, INCIDENT_TEMPLATES,
        FRAMEWORK_MINIMAL, FRAMEWORK_MODERATE, FRAMEWORK_STRICT,
    )

    model = _get_model()
    embedder = _get_embedder()

    incident = None
    for t in INCIDENT_TEMPLATES:
        if args.incident.lower() in t.name.lower():
            incident = t
            break

    if incident is None:
        print(f"Unknown incident: {args.incident}")
        print("Available incidents:")
        for t in INCIDENT_TEMPLATES:
            print(f"  - {t.name}")
        return

    frameworks_map = {
        "minimal": FRAMEWORK_MINIMAL,
        "moderate": FRAMEWORK_MODERATE,
        "strict": FRAMEWORK_STRICT,
        "all": None,
    }

    frameworks = None
    if args.framework != "all":
        fw = frameworks_map.get(args.framework)
        if fw:
            frameworks = [fw]

    wg = WarGame(
        model=model,
        embedder=embedder,
        n_ensemble=args.n_ensemble,
        num_rounds=args.rounds,
        output_dir=args.output,
    )

    report = wg.run_war_game(incident=incident, frameworks=frameworks)
    report.print_summary()


def cmd_blind_spots(args):
    """Run blind-spot finder across randomized severity/trust/investment scenarios."""
    from policylab.features.blind_spot_finder import BlindSpotFinder
    from policylab.features.war_game import (
        FRAMEWORK_MINIMAL, FRAMEWORK_MODERATE, FRAMEWORK_STRICT,
    )

    model = _get_model()
    embedder = _get_embedder()

    frameworks_map = {
        "minimal": FRAMEWORK_MINIMAL,
        "moderate": FRAMEWORK_MODERATE,
        "strict": FRAMEWORK_STRICT,
    }

    framework = frameworks_map.get(args.framework, FRAMEWORK_MODERATE)

    finder = BlindSpotFinder(
        model=model,
        embedder=embedder,
        n_scenarios=args.n_scenarios,
        n_ensemble_per_scenario=args.n_ensemble,
        num_rounds=args.rounds,
        output_dir=args.output,
    )

    report = finder.find_blind_spots(framework=framework)
    report.print_summary()


def cmd_backtest(args):
    """CLI handler for the backtest command."""
    from policylab.features.stress_tester import StressTester
    from policylab.validation.backtester import (
        Backtester, EU_AI_ACT, US_EXECUTIVE_ORDER,
    )

    model = _get_model()
    embedder = _get_embedder()

    tester = StressTester(
        model=model,
        embedder=embedder,
        n_ensemble=args.n_ensemble,
        num_rounds=args.rounds,
        output_dir=os.path.join(args.output, "stress"),
    )

    backtester = Backtester(
        stress_tester=tester,
        output_dir=args.output,
    )

    cases_map = {
        "eu-ai-act": EU_AI_ACT,
        "us-executive-order": US_EXECUTIVE_ORDER,
        "all": None,
    }

    if args.case == "all":
        reports = backtester.backtest_all()
    else:
        case = cases_map.get(args.case)
        if case is None:
            print(f"Unknown case: {args.case}")
            print("Available: eu-ai-act, us-executive-order, all")
            return
        report = backtester.backtest(case, skip_contaminated=args.skip_contaminated)
        report.print_summary()



def cmd_v2_stress_test(args):
    """Run the v2 hybrid stress test — calibrated population agents + optional LLM.

    Population-only mode (default, fast):
        python -m policylab v2-stress-test --policy "..." --description "..."

    Full hybrid mode (population + 5 LLM strategic agents):
        python -m policylab v2-stress-test --policy "..." --description "..." --llm

    The LLM agents run the government, regulator, industry association,
    civil society leader, and safety-first corporation roles. They receive
    population statistics each round and make institutional strategic decisions.
    Requires: OPENAI_API_KEY set. Cost: ~$5-15 per ensemble run.
    """
    from policylab.v2.stress_test_v2 import HybridStressTest, HybridEnsembleReport
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig

    use_llm = getattr(args, "llm", False)

    # Build model and embedder only if LLM mode requested
    llm_model = None
    llm_embedder = None
    if use_llm:
        print("[v2] LLM mode enabled — initialising language model...")
        try:
            llm_model = _get_model()
            # Try real embedder first, fall back to random
            try:
                from policylab.embedder import EmbedderModel
                llm_embedder = EmbedderModel()
            except Exception:
                from policylab.v2.simulation.llm_bridge import RandomEmbedder
                llm_embedder = RandomEmbedder()
                print("[v2] Using random embedder (install sentence-transformers for better results)")
        except Exception as e:
            print(f"[v2] LLM initialisation failed: {e}")
            print("[v2] Falling back to population-only mode.")
            use_llm = False

    tester = HybridStressTest(
        n_population=getattr(args, "n_population", 100),
        num_rounds=getattr(args, "rounds", 16),
        spillover_factor=0.5,
        output_dir=getattr(args, "output", "./results/v2_stress_test"),
    )

    # Patch HybridStressTest to inject LLM config if --llm passed
    if use_llm:
        original_run = tester.run
        def run_with_llm(policy_name, policy_description, policy_severity=None, n_ensemble=5):
            import time
            from policylab.v2.simulation.hybrid_loop import run_hybrid_simulation
            from policylab.v2.stress_test_v2 import HybridEnsembleReport
            severity = policy_severity or _detect_severity(policy_description)
            print(f"\n[v2 HYBRID] Running {n_ensemble} runs with LLM strategic agents...")
            results = []
            for i in range(n_ensemble):
                seed = 42 + i
                config = HybridSimConfig(
                    n_population=tester.n_population,
                    num_rounds=tester.num_rounds,
                    spillover_factor=tester.spillover_factor,
                    seed=seed,
                    verbose=(i == 0),
                    run_llm_strategic=True,
                    llm_model=llm_model,
                    llm_embedder=llm_embedder,
                )
                print(f"  Run {i+1}/{n_ensemble} (seed={seed}, hybrid LLM+population)...",
                      end=" ", flush=True)
                t0 = time.time()
                r = run_hybrid_simulation(policy_name, policy_description, severity, config)
                print(f"{time.time()-t0:.0f}s")
                results.append(r)
            return HybridEnsembleReport(
                policy_name=policy_name,
                policy_severity=severity,
                n_runs=n_ensemble,
                results=results,
            )
        tester.run = run_with_llm

    report = tester.run(
        policy_name=args.policy,
        policy_description=args.description,
        policy_severity=getattr(args, "severity", None),
        n_ensemble=getattr(args, "n_ensemble", 5),
    )
    print(report.summary())


def _detect_severity(desc: str) -> float:
    """Auto-detect severity from description (used by v2 CLI)."""
    d = desc.lower()
    if any(w in d for w in ["moratorium", "ban", "prohibition", "dissolution", "imprisonment"]):
        return 5.0
    elif any(w in d for w in ["criminal", "mandatory", "shutdown"]):
        return 4.0
    elif any(w in d for w in ["require", "comply", "audit", "mandatory"]):
        return 3.0
    elif any(w in d for w in ["register", "report", "disclose"]):
        return 2.0
    return 1.0


def cmd_demo(args):
    """Run a quick demo of all features."""
    from policylab.scenarios.simple_regulation import (
        build_scenario, run_simulation, save_results,
    )

    print("=" * 70)
    print("POLICYLAB — DEMO")
    print("=" * 70)
    print("\nRunning all features with minimal settings.")
    print("For production use, increase --n-ensemble to 30+.\n")

    print("\n" + "=" * 70)
    print("1/4: SIMPLE REGULATION SCENARIO")
    print("=" * 70)
    model, world_state, agents, resources, engine = build_scenario()
    data = run_simulation(model, world_state, agents, resources, engine, num_rounds=3)
    save_results(data, "./results/demo/simple")

    print("\n" + "=" * 70)
    print("2/4: STRESS TEST")
    print("=" * 70)
    from policylab.features.stress_tester import StressTester
    model = _get_model()
    embedder = _get_embedder()
    tester = StressTester(model=model, embedder=embedder, n_ensemble=3, num_rounds=3)
    report = tester.stress_test(
        policy_name="AI Registration Act",
        policy_description="Mandatory registration for AI models above 10^25 FLOPS",
        regulated_entities=["AI companies"],
        requirements=["Register models", "Safety evaluation"],
        penalties=["Up to $10M fines"],
    )
    report.print_summary()

    print("\n" + "=" * 70)
    print("3/4: WAR GAME")
    print("=" * 70)
    from policylab.features.war_game import (
        WarGame, INCIDENT_TEMPLATES, FRAMEWORK_MODERATE,
    )
    wg = WarGame(model=model, embedder=embedder, n_ensemble=2, num_rounds=3)
    report = wg.run_war_game(
        incident=INCIDENT_TEMPLATES[0],
        frameworks=[FRAMEWORK_MODERATE],
    )
    report.print_summary()

    print("\n" + "=" * 70)
    print("4/4: HISTORICAL BACKTEST")
    print("=" * 70)
    from policylab.validation.backtester import Backtester, EU_AI_ACT
    bt_tester = StressTester(model=model, embedder=embedder, n_ensemble=3, num_rounds=3)
    backtester = Backtester(stress_tester=bt_tester)
    bt_report = backtester.backtest(EU_AI_ACT)
    bt_report.print_summary()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll results saved to ./results/demo/")
    print("\nFor production runs with real LLM:")
    print("  export OPENAI_API_KEY='sk-...'")
    print("  python -m policylab.cli stress-test --n-ensemble 30")


def cmd_ingest(args) -> None:
    """Ingest a regulatory document and optionally run a simulation.

    Example:
        policylab ingest eu_ai_act.pdf --traceability
        policylab ingest sb53.pdf --api-key sk-... --model gpt-4o
        policylab ingest bill.txt --no-simulate --output-json result.json
    """
    import json as _json
    from policylab.v2.ingest.pipeline import ingest
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation

    result = ingest(
        args.file,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        verbose=True,
    )

    print()
    print(result.extraction.summary_table())

    if args.traceability:
        print()
        print(result.traceability_report())

    if args.output_json:
        data = {
            "policy_name":        str(result.spec.name),
            "severity":           result.spec.severity,
            "compute_cost_factor": result.spec.compute_cost_factor,
            "confidence":         result.confidence_summary(),
            "warnings":           result.warnings,
            "config":             {k: str(v) for k, v in result.config.items()},
        }
        with open(args.output_json, "w") as f:
            _json.dump(data, f, indent=2)
        print(f"\n[ingest] Extraction written to {args.output_json}")

    if args.no_simulate:
        return

    config_kwargs = dict(result.config)
    config_kwargs["verbose"] = True
    config_kwargs["seed"] = args.seed
    if args.rounds:
        config_kwargs["num_rounds"] = args.rounds
    if args.n_population:
        config_kwargs["n_population"] = args.n_population

    print(f"\n[ingest] Running simulation for \"{result.spec.name}\"...")
    sim = run_hybrid_simulation(
        result.spec.name, result.spec.description, result.spec.severity,
        config=HybridSimConfig(**config_kwargs),
    )
    fp = sim.final_population_summary
    fs = sim.final_stocks
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS — {result.spec.name}")
    print(f"  Severity:        {result.spec.severity:.2f} / 5.0")
    print(f"  Compliance:      {fp.get('compliance_rate',0):.0%}")
    print(f"  Relocation:      {fp.get('relocation_rate',0):.0%}")
    print(f"  Ever lobbied:    {fp.get('ever_lobbied_rate',0):.0%}")
    print(f"  AI investment:   {fs.get('ai_investment_index',0):.0f}/100")
    print(f"{'='*60}")


def main():
    """Parse args and dispatch to command handler."""
    parser = argparse.ArgumentParser(
        description="PolicyLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m policylab.cli demo
  python -m policylab.cli stress-test --policy "AI Registration" --description "..."
  python -m policylab.cli war-game --incident "Deepfake"
  python -m policylab.cli blind-spots --n-scenarios 100
  python -m policylab.cli backtest --case eu-ai-act
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    st = subparsers.add_parser("stress-test", help="Stress-test a proposed regulation")
    st.add_argument("--policy", required=True, help="Policy name")
    st.add_argument("--description", required=True, help="Policy description")
    st.add_argument("--entities", default="AI companies", help="Comma-separated regulated entities")
    st.add_argument("--requirements", default="Comply with regulation", help="Semicolon-separated requirements")
    st.add_argument("--penalties", default="Fines up to $10M", help="Semicolon-separated penalties")
    st.add_argument("--n-ensemble", type=int, default=30, help="Number of ensemble runs")
    st.add_argument("--rounds", type=int, default=8, help="Rounds per simulation")
    st.add_argument("--output", default="./results/stress_test", help="Output directory")
    st.add_argument("--revalidate", action="store_true", help="After runs, ask the LLM to review all action classifications and flag errors")

    wg = subparsers.add_parser("war-game", help="War-game an AI incident")
    wg.add_argument("--incident", required=True, help="Incident name (partial match)")
    wg.add_argument("--framework", default="all", choices=["minimal", "moderate", "strict", "all"])
    wg.add_argument("--n-ensemble", type=int, default=10, help="Ensemble runs per framework")
    wg.add_argument("--rounds", type=int, default=6, help="Rounds per simulation")
    wg.add_argument("--output", default="./results/war_game", help="Output directory")

    bs = subparsers.add_parser("blind-spots", help="Find governance blind spots")
    bs.add_argument("--framework", default="moderate", choices=["minimal", "moderate", "strict"])
    bs.add_argument("--n-scenarios", type=int, default=50, help="Number of scenario variations")
    bs.add_argument("--n-ensemble", type=int, default=5, help="Ensemble runs per scenario")
    bs.add_argument("--rounds", type=int, default=4, help="Rounds per simulation")
    bs.add_argument("--output", default="./results/blind_spots", help="Output directory")

    bt = subparsers.add_parser("backtest", help="Backtest against historical cases")
    bt.add_argument("--case", default="all", choices=["eu-ai-act", "us-executive-order", "all"])
    bt.add_argument("--n-ensemble", type=int, default=30, help="Ensemble runs")
    bt.add_argument("--rounds", type=int, default=8, help="Rounds per simulation")
    bt.add_argument("--output", default="./results/backtest", help="Output directory")
    bt.add_argument(
        "--skip-contaminated",
        action="store_true",
        default=False,
        help="Skip cases with contamination score ≥ 0.7 (policy in LLM training data)",
    )


    v2 = subparsers.add_parser(
        "v2-stress-test",
        help="Hybrid stress test: 100+ calibrated population agents (v2 engine)",
    )
    v2.add_argument("--policy", required=True, help="Policy name")
    v2.add_argument("--description", required=True, help="Policy description")
    v2.add_argument("--severity", type=float, default=None,
                    help="Policy severity 1-5 (auto-detected if omitted)")
    v2.add_argument("--n-population", type=int, default=100,
                    help="Number of population agents (default 100, min 50)")
    v2.add_argument("--n-ensemble", type=int, default=5,
                    help="Ensemble runs (default 5)")
    v2.add_argument("--rounds", type=int, default=16,
                    help="Rounds per run — 1 round = 3 months (default 16 = 4 years)")
    v2.add_argument("--output", default="./results/v2_stress_test",
                    help="Output directory")
    v2.add_argument("--llm", action="store_true",
                    help="Enable 5 LLM strategic agents alongside population agents "
                         "(requires OPENAI_API_KEY; ~$5-15/run)")
    v2.add_argument("--compare", nargs="+", metavar="POLICY_JSON",
                    help="Compare multiple policies: pass JSON files with policy specs")

    subparsers.add_parser("demo", help="Quick demo of all features")

    ig = subparsers.add_parser(
        "ingest",
        help="Ingest a regulatory document (PDF/txt/md/docx) → extract → simulate",
    )
    ig.add_argument("file", help="Document path (PDF, txt, md, or docx)")
    ig.add_argument("--api-key", default=None, metavar="KEY",
                    help="OpenAI-compatible API key (falls back to OPENAI_API_KEY env var)")
    ig.add_argument("--model", default="gpt-4o", help="LLM model (default: gpt-4o)")
    ig.add_argument("--base-url", default=None, metavar="URL",
                    help="Non-OpenAI endpoint (e.g. http://localhost:11434/v1 for Ollama)")
    ig.add_argument("--no-simulate", action="store_true",
                    help="Extract provisions only, skip simulation")
    ig.add_argument("--rounds", type=int, default=None, help="Override ingest-derived horizon")
    ig.add_argument("--n-population", type=int, default=None, help="Override population size")
    ig.add_argument("--output-json", default=None, metavar="PATH",
                    help="Write extraction JSON to file")
    ig.add_argument("--traceability", action="store_true",
                    help="Print full parameter derivation report")
    ig.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "stress-test": cmd_stress_test,
        "war-game": cmd_war_game,
        "blind-spots": cmd_blind_spots,
        "backtest": cmd_backtest,
        "demo": cmd_demo,
        "v2-stress-test": cmd_v2_stress_test,
        "ingest": cmd_ingest,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
