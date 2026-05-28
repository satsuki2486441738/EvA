import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { motion } from "framer-motion";
import {
  BookOpen,
  Check,
  Clipboard,
  Clock,
  Code2,
  Database,
  ExternalLink,
  Layers,
  Package,
  Play,
  Sparkles
} from "lucide-react";
import "./styles.css";

const actionLinks = [
  { label: "Paper", icon: BookOpen, href: "https://arxiv.org/abs/2603.27667" },
  { label: "Code", icon: Code2, href: "https://github.com/satsuki2486441738/EvA" },
  {
    label: "Models",
    icon: Package,
    links: [
      { label: "Hugging Face", href: "" },
      { label: "ModelScope", href: "" }
    ]
  },
  {
    label: "Dataset",
    icon: Database,
    links: [
      {
        label: "Hugging Face",
        href: "https://huggingface.co/datasets/SatsukiVie/EvidenceFirst-Audio"
      },
      {
        label: "ModelScope",
        href: "https://www.modelscope.cn/datasets/XinyuanXie/EvidenceFirst-Audio"
      }
    ]
  }
];

const results = [
  {
    model: "Qwen2.5-Omni-7B",
    values: ["72.06", "67.36", "52.80", "61.43", "39.96", "69.91", "80.38"]
  },
  {
    model: "Audio-Flamingo-3",
    values: ["73.90", "71.50", "59.23", "61.64", "45.63", "77.86", "75.57"]
  },
  {
    model: "Step-Audio-2-mini",
    values: ["66.54", "68.58", "54.58", "61.01", "44.36", "78.32", "83.24"]
  },
  {
    model: "Audio-Reasoner",
    values: ["65.44", "63.39", "44.34", "38.23", "40.73", "68.38", "--"]
  },
  {
    model: "R1-AQA",
    values: ["72.06", "62.87", "48.75", "50.19", "41.68", "71.94", "76.30"]
  },
  {
    model: "Kimi-Audio-7B-Instruct",
    values: ["66.18", "59.59", "56.79", "58.72", "45.47", "71.85", "86.17"]
  },
  {
    model: "EvA (Kimi-Audio)",
    values: ["77.57", "64.17", "59.79", "59.45", "47.52", "75.41", "87.04"],
    gains: ["+11.39", "+4.58", "+3.00", "+0.73", "+2.05", "+3.56", "+0.87"],
    ours: true
  }
];

const bibtex = `@misc{xie2026evaevidencefirstaudiounderstanding,
      title={EvA: An Evidence-First Audio Understanding Paradigm for LALMs}, 
      author={Xinyuan Xie and Shunian Chen and Zhiheng Liu and Yuhao Zhang and Zhiqiang Lv and Liyin Liang and Benyou Wang},
      year={2026},
      eprint={2603.27667},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.27667}, 
}`;

function Section({ id, eyebrow, title, children, className = "" }) {
  return (
    <section id={id} className={`px-5 py-20 sm:px-8 ${className}`}>
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.55, ease: "easeOut" }}
        className="mx-auto max-w-7xl"
      >
        {eyebrow && (
          <p className="mb-3 text-center text-sm font-semibold uppercase tracking-[0.18em] text-indigo-600">
            {eyebrow}
          </p>
        )}
        {title && (
          <h2 className="mx-auto mb-10 max-w-3xl text-center text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
            {title}
          </h2>
        )}
        {children}
      </motion.div>
    </section>
  );
}

function ActionButton({ href, label, icon: Icon }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="group inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-800 shadow-sm transition hover:-translate-y-0.5 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
    >
      <Icon className="h-4 w-4 transition group-hover:scale-110" />
      {label}
    </a>
  );
}

function LinkMenu({ label, icon: Icon, links }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setOpen(false);
      }
    }

    document.addEventListener("pointerdown", handleClickOutside);
    return () => document.removeEventListener("pointerdown", handleClickOutside);
  }, []);

  return (
    <div ref={menuRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-800 shadow-sm transition hover:-translate-y-0.5 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
      >
        <Icon className="h-4 w-4 transition" />
        {label}
      </button>
      {open && (
        <div className="absolute left-1/2 z-20 mt-2 w-48 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-1 text-left opacity-100 shadow-lg">
        {links.map((link) =>
          link.href ? (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noreferrer"
              onClick={() => setOpen(false)}
              className="flex items-center justify-between rounded-md px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-indigo-50 hover:text-indigo-700"
            >
              {link.label}
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          ) : (
            <span
              key={link.label}
              className="flex cursor-not-allowed items-center justify-between rounded-md px-3 py-2 text-sm font-medium text-slate-400"
            >
              {link.label}
              <span className="text-xs">Soon</span>
            </span>
          )
        )}
        </div>
      )}
    </div>
  );
}

function Hero() {
  return (
    <header className="relative overflow-hidden border-b border-slate-200 bg-white px-5 py-20 sm:px-8 lg:py-24">
      <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-indigo-600 via-sky-500 to-emerald-500" />
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.65, ease: "easeOut" }}
        className="mx-auto max-w-5xl text-center"
      >
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-sm font-medium text-indigo-700">
          <Sparkles className="h-4 w-4" />
          Evidence-first audio understanding for LALMs
        </div>
        <h1 className="mx-auto max-w-7xl text-balance text-4xl font-bold tracking-tight text-slate-950 sm:text-5xl lg:text-[3.35rem]">
          EvA: An Evidence-First Audio Understanding Paradigm for LALMs
        </h1>
        <p className="mx-auto mt-6 max-w-5xl text-base leading-7 text-slate-600 sm:text-lg">
          Xinyuan Xie, Shunian Chen, Zhiheng Liu, Yuhao Zhang, Zhiqiang Lv,
          Liyin Liang, Benyou Wang
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          {actionLinks.map((link) => (
            link.links ? (
              <LinkMenu key={link.label} {...link} />
            ) : (
              <ActionButton key={link.label} {...link} />
            )
          ))}
        </div>
        <p className="mx-auto mt-10 max-w-3xl rounded-lg border border-slate-200 bg-slate-50 px-6 py-5 text-left text-base leading-8 text-slate-700 shadow-soft sm:text-center">
          Large Audio Language Models (LALMs) struggle in complex acoustic
          scenes due to weak perceptual grounding. EvA introduces an
          evidence-first dual-path architecture to break this bottleneck,
          achieving competitive open-source performance on MMAU, MMAR, and MMSU
          benchmarks.
        </p>
      </motion.div>
    </header>
  );
}

function Motivation() {
  const bars = [
    {
      label: "Perception",
      model: 42.8,
      human: 91.2,
      note: "largest gap"
    },
    {
      label: "Reasoning",
      model: 73.5,
      human: 86.8,
      note: "narrower gap"
    }
  ];

  return (
    <Section id="motivation" eyebrow="Motivation" title="The Evidence Bottleneck">
      <div className="grid items-center gap-10 md:grid-cols-2">
        <div className="space-y-5 text-lg leading-8 text-slate-700">
          <p>
            Complex audio understanding fails when a model misses the acoustic
            evidence needed for downstream reasoning. The central bottleneck is
            often not the language model's ability to infer, but its ability to
            extract, localize, and preserve perceptual cues from dense sound
            scenes.
          </p>
          <p>
            EvA treats evidence as the first-class interface between audio
            perception and language reasoning. It strengthens multi-level
            acoustic grounding before asking the LALM to answer, explain, or
            compare.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-soft">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-950">
                Model vs. Human Performance
              </h3>
              <p className="text-sm text-slate-500">
                Diagnostic split by capability
              </p>
            </div>
            <div className="flex gap-4 text-xs font-medium text-slate-600">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-indigo-600" />
                Model
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                Human
              </span>
            </div>
          </div>
          <div className="space-y-7">
            {bars.map((bar) => (
              <div key={bar.label}>
                <div className="mb-2 flex items-end justify-between">
                  <div>
                    <p className="font-semibold text-slate-900">{bar.label}</p>
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      {bar.note}
                    </p>
                  </div>
                  <p className="text-sm text-slate-500">
                    {bar.model}% vs {bar.human}%
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="h-4 overflow-hidden rounded-full bg-slate-100">
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${bar.model}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8 }}
                      className="h-full rounded-full bg-indigo-600"
                    />
                  </div>
                  <div className="h-4 overflow-hidden rounded-full bg-slate-100">
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${bar.human}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, delay: 0.1 }}
                      className="h-full rounded-full bg-emerald-500"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

function ArchitecturePanel() {
  return (
    <div>
      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-soft">
        <div className="border-b border-slate-200 bg-slate-100 px-5 py-3 text-sm font-medium text-slate-600">
          Figure 2: Model Architecture
        </div>
        <div className="bg-white p-4 sm:p-6">
          <img
            src="./assets/architecture.png"
            alt="EvA architecture diagram"
            className="mx-auto w-full max-w-5xl rounded-md object-contain"
          />
        </div>
      </div>
      <div className="mt-8 grid gap-5 md:grid-cols-2">
        <FeatureCard
          icon={Layers}
          title="Hierarchical Evidence Aggregation"
          text="Fusing multi-scale acoustic cues across frequencies and depths."
        />
        <FeatureCard
          icon={Clock}
          title="Time-Aware Inject-and-Add Fusion"
          text="Non-compressive alignment to the Whisper timeline preserving temporal fidelity."
        />
      </div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, text }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-md bg-indigo-50 text-indigo-700">
        <Icon className="h-5 w-5" />
      </div>
      <h3 className="text-lg font-semibold text-slate-950">{title}</h3>
      <p className="mt-2 leading-7 text-slate-600">{text}</p>
    </div>
  );
}

function DatasetPanel() {
  return (
    <div>
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-soft">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-indigo-600">
            Dataset Interface
          </p>
          <h3 className="mt-2 text-2xl font-bold text-slate-950">
            Audio-grounded QA Explorer
          </h3>
          <p className="mt-4 leading-7 text-slate-600">
            A reserved presentation panel for EvA-Perception examples, designed
            to show audio evidence, captions, and question-answer annotations
            once the final dataset samples are ready.
          </p>
          <div className="mt-6 grid grid-cols-3 gap-3">
            {[
              ["Audio", "clip"],
              ["Caption", "evidence"],
              ["QA", "pairs"]
            ].map(([label, value]) => (
              <div key={label} className="rounded-md border border-slate-200 bg-slate-50 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {label}
                </p>
                <p className="mt-1 text-sm font-semibold text-slate-900">{value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white shadow-soft">
          <div className="border-b border-slate-200 px-5 py-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-slate-950">Sample Preview</p>
                <p className="text-sm text-slate-500">Placeholder layout</p>
              </div>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                Pending data
              </span>
            </div>
          </div>
          <div className="space-y-5 p-5">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  aria-label="Audio preview placeholder"
                  className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-slate-300 text-white"
                  disabled
                >
                  <Play className="ml-0.5 h-5 w-5" />
                </button>
                <div className="min-w-0 flex-1">
                  <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                    <div className="h-full w-2/5 rounded-full bg-slate-300" />
                  </div>
                  <div className="mt-2 flex justify-between text-xs font-medium text-slate-400">
                    <span>0:00</span>
                    <span>0:15</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <span className="h-2 w-24 rounded-full bg-slate-200" />
              <span className="h-2 flex-1 rounded-full bg-slate-100" />
            </div>

            <div className="rounded-lg border border-slate-200 p-4">
              <div className="mb-3 h-2 w-40 rounded-full bg-slate-200" />
              <div className="space-y-2">
                <div className="h-2 rounded-full bg-slate-100" />
                <div className="h-2 w-11/12 rounded-full bg-slate-100" />
                <div className="h-2 w-4/5 rounded-full bg-slate-100" />
              </div>
            </div>

            {[0, 1, 2].map((item) => (
              <div key={item} className="rounded-lg border border-slate-200 p-4">
                <div className="mb-3 h-2 w-3/4 rounded-full bg-slate-200" />
                <div className="h-2 w-1/2 rounded-full bg-slate-100" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Artifacts() {
  const [activeTab, setActiveTab] = useState("architecture");
  const tabs = [
    { id: "architecture", label: "Architecture", icon: Layers },
    { id: "dataset", label: "Dataset Preview", icon: Database }
  ];

  return (
    <Section id="artifacts" eyebrow="Method & Data" title="Research Artifacts" className="bg-slate-50">
      <div className="mb-8 flex justify-center">
        <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex min-h-10 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
                  isActive
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-950"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
      {activeTab === "architecture" ? <ArchitecturePanel /> : <DatasetPanel />}
    </Section>
  );
}

function Results() {
  return (
    <Section id="results" eyebrow="Experiments" title="Benchmark Performance" className="bg-slate-50">
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-soft">
        <table className="min-w-full divide-y divide-slate-200 text-left">
          <thead className="bg-slate-100">
            <tr>
              <th rowSpan="2" className="px-5 py-4 text-sm font-semibold uppercase tracking-wide text-slate-600">
                Model
              </th>
              <th colSpan="2" className="border-l border-slate-200 px-5 py-3 text-center text-sm font-semibold uppercase tracking-wide text-slate-600">
                MMAU-mini(clean)
              </th>
              <th colSpan="2" className="border-l border-slate-200 px-5 py-3 text-center text-sm font-semibold uppercase tracking-wide text-slate-600">
                MMAR
              </th>
              <th colSpan="2" className="border-l border-slate-200 px-5 py-3 text-center text-sm font-semibold uppercase tracking-wide text-slate-600">
                MMSU
              </th>
              <th rowSpan="2" className="border-l border-slate-200 px-5 py-4 text-sm font-semibold uppercase tracking-wide text-slate-600">
                CochlScene
              </th>
            </tr>
            <tr>
              {["Perc.", "Reas.", "Perc.", "Reas.", "Perc.", "Reas."].map((header, index) => (
                <th
                  key={`${header}-${index}`}
                  className={`px-5 py-3 text-center text-xs font-semibold uppercase tracking-wide text-slate-500 ${
                    index % 2 === 0 ? "border-l border-slate-200" : ""
                  }`}
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {results.map((row) => {
              const isOurs = row.ours;
              return (
                <tr
                  key={row.model}
                  className={isOurs ? "bg-indigo-50/70 font-bold text-slate-950" : "text-slate-700"}
                >
                  <td className="whitespace-nowrap px-5 py-4 font-semibold">{row.model}</td>
                  {row.values.map((cell, index) => (
                    <td
                      key={cell + index}
                      className={`whitespace-nowrap px-5 py-4 text-center ${
                        [0, 2, 4, 6].includes(index) ? "border-l border-slate-200" : ""
                      }`}
                    >
                      <span>{cell}</span>
                      {row.gains?.[index] && (
                        <span className="ml-1 text-xs font-bold text-emerald-700">
                          {row.gains[index]}
                        </span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-center text-sm leading-6 text-slate-500">
        Unified zero-shot setting. MMAU-mini(clean) denotes the decontaminated split.
      </p>
    </Section>
  );
}

function Citation() {
  const [copied, setCopied] = useState(false);

  async function copyBibtex() {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(bibtex);
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <footer className="bg-slate-950 px-5 py-16 text-white sm:px-8">
      <div className="mx-auto max-w-5xl">
        <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-indigo-300">
              Citation
            </p>
            <h2 className="mt-2 text-3xl font-bold tracking-tight">BibTeX</h2>
          </div>
          <button
            type="button"
            onClick={copyBibtex}
            className="inline-flex min-h-10 items-center justify-center gap-2 rounded-md bg-white px-3 py-2 text-sm font-semibold text-slate-950 transition hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:ring-offset-2 focus:ring-offset-slate-950"
          >
            {copied ? <Check className="h-4 w-4" /> : <Clipboard className="h-4 w-4" />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-900 p-5 text-sm leading-7 text-slate-100 shadow-soft">
          <code>{bibtex}</code>
        </pre>
      </div>
    </footer>
  );
}

function App() {
  return (
    <main className="min-h-screen bg-white text-slate-900">
      <Hero />
      <Motivation />
      <Artifacts />
      <Results />
      <Citation />
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
