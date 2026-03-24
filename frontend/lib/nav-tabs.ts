import { BarChart2, Scale, Upload, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface NavTab {
  href: "/analyze" | "/upload" | "/compare" | "/influence";
  label: string;
  icon: LucideIcon;
}

export const NAV_TABS: NavTab[] = [
  { href: "/analyze",   label: "Analyze",  icon: BarChart2 },
  { href: "/upload",    label: "Upload",   icon: Upload    },
  { href: "/compare",   label: "Compare",  icon: Scale     },
  { href: "/influence", label: "Influence",icon: Zap       },
];
