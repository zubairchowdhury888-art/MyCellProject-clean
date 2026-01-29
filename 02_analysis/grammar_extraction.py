"""
Grammar extraction for quantised fungal spike sequences.

Reads *_symseq.txt from 03_results/grammar_sequences,
runs a Sequitur-like grammar induction per sequence,
and writes dataset-level grammar metrics to 03_results/grammar_metrics.json.
"""

import os
import json
from collections import Counter

# --- Paths (relative to MyCellProject root) ---

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # MyCellProject/
GRAMMAR_SEQ_DIR = os.path.join(BASE_DIR, "03_results", "grammar_sequences")
OUT_DIR = os.path.join(BASE_DIR, "03_results")
OUT_JSON = os.path.join(OUT_DIR, "grammar_metrics.json")


# --- Core grammar data structures ---

class Rule:
    __slots__ = ("name", "rhs")

    def __init__(self, name):
        self.name = name
        self.rhs = []  # list of symbols (terminals or nonterminals)


class SequiturGrammar:
    """
    Minimal Sequitur-style grammar inducer.

    Enforces:
    - Digram uniqueness: no pair of adjacent symbols occurs more than once.
    - Rule utility: every nonterminal is used at least twice.[web:21][web:22]
    """

    def __init__(self):
        self.rules = {}              # name -> Rule
        self.start = Rule("S")
        self.rules[self.start.name] = self.start
        self.digram_index = {}       # (sym_i, sym_j) -> (rule_name, position)
        self.next_rule_id = 1

    # --- public API ---

    def ingest_sequence(self, seq):
        """Feed a list of terminal symbols into the grammar."""
        for sym in seq:
            self._append_symbol(self.start, sym)

    def compute_metrics(self, seq_length):
        """Compute dataset-level metrics from the induced grammar."""
        # Rule counts
        n_rules_total = len(self.rules)
        n_nonterminals = n_rules_total - 1  # exclude S

        # Grammar size and compression ratio.[web:5]
        grammar_size = sum(len(rule.rhs) for rule in self.rules.values())
        compression_ratio = seq_length / grammar_size if grammar_size > 0 else 1.0
        saved_symbols = seq_length - grammar_size

        # Depth metrics (cycle-safe).
        depth_stats = self._compute_depth_stats()
        depth_mean = depth_stats["depth_mean"]
        depth_max = depth_stats["depth_max"]
        depth_std = depth_stats["depth_std"]

        # Motif (rule) span and usage statistics.[web:24][web:31]
        motif_stats = self._compute_motif_stats()
        span_mean = motif_stats["span_length_mean"]
        span_max = motif_stats["span_length_max"]
        usage_mean = motif_stats["usage_mean"]
        usage_max = motif_stats["usage_max"]

        return {
            "seq_length": seq_length,
            "n_rules_total": n_rules_total,
            "n_nonterminals": n_nonterminals,
            "grammar_size": grammar_size,
            "compression_ratio": compression_ratio,
            "saved_symbols": saved_symbols,
            "depth_mean": depth_mean,
            "depth_max": depth_max,
            "depth_std": depth_std,
            "span_length_mean": span_mean,
            "span_length_max": span_max,
            "usage_mean": usage_mean,
            "usage_max": usage_max,
        }

    # --- Sequitur internals ---

    def _append_symbol(self, rule, sym):
        rule.rhs.append(sym)
        self._check_last_digram(rule.name)

    def _check_last_digram(self, rule_name):
        rule = self.rules[rule_name]
        if len(rule.rhs) < 2:
            return
        i = len(rule.rhs) - 2
        digram = (rule.rhs[i], rule.rhs[i + 1])

        if digram in self.digram_index:
            other_rule, other_pos = self.digram_index[digram]
            if other_rule == rule_name and other_pos == i:
                return
            self._substitute_digram(digram, other_rule, other_pos, rule_name, i)
        else:
            self.digram_index[digram] = (rule_name, i)

    def _substitute_digram(self, digram, rule_a, pos_a, rule_b, pos_b):
        # Try to reuse an existing rule whose RHS is exactly this digram.
        nt_name = self._find_existing_rule_for_digram(digram)
        if nt_name is None:
            nt_rule = self._new_rule()
            nt_rule.rhs = [digram[0], digram[1]]
            self.rules[nt_rule.name] = nt_rule
            nt_name = nt_rule.name

        # Replace digram occurrence in both rules.
        self._replace_digram(rule_a, pos_a, nt_name)
        self._replace_digram(rule_b, pos_b, nt_name)

        # Enforce rule utility (remove rules used < 2 times).[web:21][web:25]
        self._enforce_rule_utility()

    def _replace_digram(self, rule_name, pos, nt_name):
        rule = self.rules[rule_name]
        # Remove digrams overlapping this region from index.
        for i in range(max(0, pos - 1), min(len(rule.rhs) - 1, pos + 2)):
            d = (rule.rhs[i], rule.rhs[i + 1])
            if d in self.digram_index and self.digram_index[d] == (rule_name, i):
                del self.digram_index[d]
        # Replace two symbols with one nonterminal.
        del rule.rhs[pos:pos + 2]
        rule.rhs.insert(pos, nt_name)
        # Reinsert affected digrams.
        for i in range(max(0, pos - 1), min(len(rule.rhs) - 1, pos + 1)):
            d = (rule.rhs[i], rule.rhs[i + 1])
            self.digram_index[d] = (rule_name, i)

    def _find_existing_rule_for_digram(self, digram):
        for name, rule in self.rules.items():
            if name == "S":
                continue
            if len(rule.rhs) == 2 and tuple(rule.rhs) == digram:
                return name
        return None

    def _new_rule(self):
        name = f"R{self.next_rule_id}"
        self.next_rule_id += 1
        return Rule(name)

    def _enforce_rule_utility(self):
        # Count nonterminal uses.
        use_counts = Counter()
        for rname, rule in self.rules.items():
            for sym in rule.rhs:
                if sym in self.rules:  # nonterminal
                    use_counts[sym] += 1

        # Collect rules used fewer than twice (excluding S).[web:21][web:25]
        to_remove = [name for name in self.rules
                     if name != "S" and use_counts.get(name, 0) < 2]

        if not to_remove:
            return

        # Inline each such rule where it appears.
        for dead in to_remove:
            body = self.rules[dead].rhs
            for rname, rule in self.rules.items():
                new_rhs = []
                for sym in rule.rhs:
                    if sym == dead:
                        new_rhs.extend(body)
                    else:
                        new_rhs.append(sym)
                rule.rhs = new_rhs
            del self.rules[dead]

        # Rebuild digram index conservatively.
        self.digram_index.clear()
        for rname, rule in self.rules.items():
            for i in range(len(rule.rhs) - 1):
                d = (rule.rhs[i], rule.rhs[i + 1])
                self.digram_index[d] = (rname, i)

    # --- depth & motif metrics ---

    def _compute_depth_stats(self):
        depths = []
        memo = {}
        for name in self.rules:
            if name == "S":
                continue
            d = self._expansion_depth(name, memo)
            depths.append(d)
        if not depths:
            return {"depth_mean": 0.0, "depth_max": 0, "depth_std": 0.0}
        mean = sum(depths) / len(depths)
        max_d = max(depths)
        var = sum((d - mean) ** 2 for d in depths) / len(depths)
        return {
            "depth_mean": mean,
            "depth_max": max_d,
            "depth_std": var ** 0.5,
        }

    def _expansion_depth(self, rule_name, memo, stack=None):
        """Maximum nesting depth; cuts off on cycles to avoid infinite recursion."""
        if stack is None:
            stack = set()
        if rule_name in memo:
            return memo[rule_name]
        if rule_name in stack:
            # Cycle detected: treat as depth 0.
            memo[rule_name] = 0
            return 0
        stack.add(rule_name)
        rule = self.rules[rule_name]
        if not rule.rhs:
            memo[rule_name] = 0
            stack.remove(rule_name)
            return 0
        depths = []
        for sym in rule.rhs:
            if sym in self.rules:
                depths.append(1 + self._expansion_depth(sym, memo, stack))
            else:
                depths.append(1)
        d = max(depths)
        memo[rule_name] = d
        stack.remove(rule_name)
        return d

    def _compute_motif_stats(self):
        # Nonterminal usage counts.
        usage_counts = Counter()
        for rname, rule in self.rules.items():
            for sym in rule.rhs:
                if sym in self.rules:
                    usage_counts[sym] += 1

        # Span lengths of each nonterminal (in terminals).[web:31]
        span_lengths = []
        memo = {}
        for name in self.rules:
            if name == "S":
                continue
            span = self._terminal_span_length(name, memo)
            span_lengths.append(span)

        if span_lengths:
            span_mean = sum(span_lengths) / len(span_lengths)
            span_max = max(span_lengths)
        else:
            span_mean = 0.0
            span_max = 0

        if usage_counts:
            usage_mean = sum(usage_counts.values()) / len(usage_counts)
            usage_max = max(usage_counts.values())
        else:
            usage_mean = 0.0
            usage_max = 0

        return {
            "span_length_mean": span_mean,
            "span_length_max": span_max,
            "usage_mean": usage_mean,
            "usage_max": usage_max,
        }

    def _terminal_span_length(self, rule_name, memo, stack=None):
        """Total number of terminals in the full expansion; cycle-safe."""
        if stack is None:
            stack = set()
        if rule_name in memo:
            return memo[rule_name]
        if rule_name in stack:
            # Cycle detected; treat as contributing zero extra span.
            memo[rule_name] = 0
            return 0
        stack.add(rule_name)
        rule = self.rules[rule_name]
        total = 0
        for sym in rule.rhs:
            if sym in self.rules:
                total += self._terminal_span_length(sym, memo, stack)
            else:
                total += 1
        memo[rule_name] = total
        stack.remove(rule_name)
        return total


# --- utilities and driver ---

def load_symbol_sequence(path):
    """Load a text file containing a single line of symbols, e.g. 'ABCDEA…'."""
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    return list(s)


def process_all_sequences():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for fname in os.listdir(GRAMMAR_SEQ_DIR):
        if not fname.endswith(".txt"):
            continue
        dataset_name = os.path.splitext(fname)[0]
        path = os.path.join(GRAMMAR_SEQ_DIR, fname)

        seq = load_symbol_sequence(path)
        N = len(seq)

        grammar = SequiturGrammar()
        grammar.ingest_sequence(seq)
        metrics = grammar.compute_metrics(N)

        results[dataset_name] = metrics

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote grammar metrics for {len(results)} datasets to {OUT_JSON}")


if __name__ == "__main__":
    process_all_sequences()