import { TopNav } from "@/components/layout/TopNav";
import { MobileNav } from "@/components/layout/MobileNav";

interface Props {
  children: React.ReactNode;
}

export function AppShell({ children }: Props) {
  return (
    <div className="min-h-screen" style={{ background: "var(--cream-50)" }}>
      <TopNav />
      <main className="pb-20 lg:pb-8">{children}</main>
      <MobileNav />
    </div>
  );
}
