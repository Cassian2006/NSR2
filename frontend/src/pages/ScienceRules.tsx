import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

type RuleCard = {
  title: string;
  status: "已接入" | "部分接入" | "文档约束";
  authority: { label: string; href: string }[];
  summary: string;
  projectUse: string[];
  boundaries?: string[];
};

const RULE_SECTIONS: Array<{ heading: string; intro: string; rules: RuleCard[] }> = [
  {
    heading: "船舶适航规则",
    intro: "这部分规则直接影响规划可行域或风险权重，是当前系统里最接近商业化船型约束的一层。",
    rules: [
      {
        title: "IMO Polar Code 极地船舶类别",
        status: "部分接入",
        authority: [{ label: "IMO Polar Code", href: "https://www.imo.org/en/ourwork/safety/pages/polar-code.aspx" }],
        summary: "Polar Code 用 A/B/C 类别描述船舶在极地水域的适航范围和操作要求。",
        projectUse: [
          "在 vessel profile 中记录 polar_category，用于解释当前船型的极地适航级别。",
          "Custom Vessel 支持用户录入 polar_category，规划结果会把该字段写回 explain。",
        ],
        boundaries: ["当前还没有把 A/B/C 直接映射成独立的航速或操纵性能模型，主要用于规则解释和船型边界记录。"],
      },
      {
        title: "IACS Polar Class / Ice Class 冰级能力",
        status: "已接入",
        authority: [
          { label: "IACS Unified Requirements", href: "https://iacs.org.uk/publications/unified-requirements/" },
          { label: "IACS UR I (Polar Class)", href: "https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/2022/02/12082657/uri-1.pdf" },
        ],
        summary: "冰级决定船体结构和冰区操作能力，不能只作为展示标签。",
        projectUse: [
          "内置 vessel profile 为不同船型设定 ice_class、max_ice_conc、max_ice_thickness_m。",
          "规划阶段会把超出冰浓度/冰厚阈值的格点直接封锁，作为连续风险之外的硬约束。",
          "ice_risk_multiplier 会提高或降低 ice 通道对连续风险代价的影响强度。",
        ],
      },
      {
        title: "最小安全水深 / 吃水约束",
        status: "已接入",
        authority: [{ label: "项目内部 bathymetry 约束", href: "https://www.copernicus.eu/" }],
        summary: "吃水和最小安全水深是最直接的可航性约束之一，必须进入 blocked 判定而不是只做展示。",
        projectUse: [
          "用户可输入 draft_m 与 min_safe_depth_m。",
          "规划阶段会根据 bathy 通道把过浅格点直接当作不可航区，与连续风险热力图并列生效。",
        ],
        boundaries: ["当前系统没有进一步建模 under-keel clearance、潮汐变化和港口级通航限制。"],
      },
    ],
  },
  {
    heading: "环境数据规则",
    intro: "这部分规则说明环境图层的来源、变量语义和在系统中的角色，避免把数据层误说成黑箱输入。",
    rules: [
      {
        title: "Copernicus Marine 海浪与风场变量",
        status: "已接入",
        authority: [
          {
            label: "Copernicus wave parameters",
            href: "https://help.marine.copernicus.eu/en/articles/6175153-how-to-describe-the-wave-height-and-wave-period-parameters",
          },
          {
            label: "Copernicus direction conventions",
            href: "https://help.marine.copernicus.eu/en/articles/5046685-which-is-the-direction-conventions-of-currents-wave-and-wind-for-copernicus-marine-products",
          },
        ],
        summary: "wave_hs、wind_u10、wind_v10 等变量的单位和方向约定必须来自权威数据定义。",
        projectUse: [
          "风浪图层和风险场都基于 Copernicus 变量定义。",
          "Stormglass 实时接入会把返回字段归一化成项目内部一致的风浪字段。",
        ],
      },
      {
        title: "AIS 的科学定位",
        status: "已接入",
        authority: [
          { label: "IMO AIS", href: "https://www.imo.org/en/OurWork/safety/navigation/ais.aspx" },
          { label: "USCG AIS overview", href: "https://www.navcen.uscg.gov/automatic-identification-system-overview" },
        ],
        summary: "AIS 是船舶识别与历史交通先验，不是安全真值。",
        projectUse: [
          "AIS 已从风险场中剥离，不再直接抬高 U-Net 风险。",
          "AIS 历史走廊偏好改成显式开关，默认关闭，只在用户主动开启时参与路径偏好。",
          "AIS 展示层和连续风险层分开渲染，避免把历史交通误读成实时风险。",
        ],
      },
    ],
  },
  {
    heading: "模型与规划方法",
    intro: "这部分规则说明模型和规划器在系统中的角色，避免把深度学习输出误说成物理真值。",
    rules: [
      {
        title: "U-Net 风险先验",
        status: "已接入",
        authority: [{ label: "U-Net paper", href: "https://arxiv.org/abs/1505.04597" }],
        summary: "U-Net 负责从多通道环境栅格中提取 safe / caution / blocked 与连续概率先验。",
        projectUse: [
          "系统当前以 U-Net 概率先验、uncertainty 和环境因子共同构造连续风险热力图。",
          "U-Net 分区层现在主要作为辅助轮廓和解释层，不再等同于最终风险表达。",
        ],
        boundaries: [
          "项目把 U-Net 输出视为风险代理或空间先验，而不是物理真实风险场。",
          "blocked / caution 仍然存在，但它们现在主要承担硬约束或边界解释角色，不代表完整风险定义。",
        ],
      },
      {
        title: "A* / D* Lite 路径搜索",
        status: "已接入",
        authority: [{ label: "D* Lite paper", href: "https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf" }],
        summary: "规划器在成本场上求低成本可行路径，而不是只找几何最短线。",
        projectUse: [
          "A*、Any-Angle、Hybrid A* 和 D* Lite 共同服务于静态和动态规划。",
          "连续风险场负责提供主要代价梯度，blocked / 浅水 / 超冰况阈值提供硬约束边界。",
          "风险模式、最小安全水深、冰况阈值、AIS 走廊偏好都会转化为搜索代价或封锁条件。",
        ],
      },
    ],
  },
  {
    heading: "当前表述边界",
    intro: "这些是项目在答辩和文档中必须保持克制的地方，避免过度宣称。",
    rules: [
      {
        title: "哪些说法能讲，哪些不能讲",
        status: "文档约束",
        authority: [],
        summary: "系统应该表述为“多源环境风险代理 + 规则约束 + 路径搜索”，而不是单模型给出绝对最优答案。",
        projectUse: [
          "可以说：系统基于权威环境数据、规则约束和深度学习风险先验生成低成本可行航线。",
          "不要说：AIS 代表安全航道、U-Net 直接预测真实风险、系统给出绝对最优航线。",
          "也不要把当前系统简化描述成只有 caution 和 blocked 两级风险，这会和连续风险主显示冲突。",
        ],
      },
    ],
  },
];

const statusTone: Record<RuleCard["status"], string> = {
  已接入: "bg-emerald-100 text-emerald-800 border-emerald-200",
  部分接入: "bg-amber-100 text-amber-800 border-amber-200",
  文档约束: "bg-slate-100 text-slate-700 border-slate-200",
};

export default function ScienceRules() {
  return (
    <div className="min-h-full bg-slate-50 px-4 py-6 md:px-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <Card className="border-slate-200 bg-white">
          <CardHeader className="space-y-2">
            <CardTitle className="text-2xl text-slate-900">科学规则与权威依据</CardTitle>
            <p className="text-sm text-slate-600">
              这一页记录 NSR2 当前在数据、模型、船型和规划层面实际采用的科学规则，以及每条规则对应的权威来源、系统接法和边界。
            </p>
          </CardHeader>
        </Card>

        {RULE_SECTIONS.map((section) => (
          <Card key={section.heading} className="border-slate-200 bg-white">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg text-slate-900">{section.heading}</CardTitle>
              <p className="text-sm text-slate-600">{section.intro}</p>
            </CardHeader>
            <CardContent className="space-y-3">
              {section.rules.map((rule) => (
                <details key={rule.title} className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
                    <div>
                      <div className="font-medium text-slate-900">{rule.title}</div>
                      <div className="mt-1 text-xs text-slate-600">{rule.summary}</div>
                    </div>
                    <span className={`shrink-0 rounded-full border px-2 py-1 text-xs ${statusTone[rule.status]}`}>{rule.status}</span>
                  </summary>
                  <div className="mt-4 space-y-3 text-sm text-slate-700">
                    {rule.authority.length ? (
                      <div>
                        <div className="mb-1 font-medium text-slate-900">权威来源</div>
                        <ul className="list-disc pl-5">
                          {rule.authority.map((source) => (
                            <li key={source.href}>
                              <a href={source.href} target="_blank" rel="noreferrer" className="text-blue-700 underline underline-offset-2">
                                {source.label}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    <div>
                      <div className="mb-1 font-medium text-slate-900">项目中的接法</div>
                      <ul className="list-disc pl-5">
                        {rule.projectUse.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                    {rule.boundaries?.length ? (
                      <div>
                        <div className="mb-1 font-medium text-slate-900">边界说明</div>
                        <ul className="list-disc pl-5">
                          {rule.boundaries.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                </details>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
