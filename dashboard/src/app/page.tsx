import { redirect } from "next/navigation";

// The Decision Cockpit is the product's front door (the engine surface).
// The news/research app remains reachable from the cockpit + the nav bar.
export default function Home() {
  redirect("/cockpit");
}
