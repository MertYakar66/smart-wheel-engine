import { redirect } from "next/navigation";

// The Decision Cockpit is the product's front door (the engine surface);
// /portfolio and /terminal are reachable from its cross-page tabs.
export default function Home() {
  redirect("/cockpit");
}
