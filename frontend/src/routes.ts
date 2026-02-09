import { createBrowserRouter } from "react-router";
import ScenarioSelector from "./pages/ScenarioSelector";
import MapWorkspace from "./pages/MapWorkspace";
import ExportReport from "./pages/ExportReport";
import Layout from "./components/Layout";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: ScenarioSelector },
      { path: "workspace", Component: MapWorkspace },
      { path: "export", Component: ExportReport },
    ],
  },
]);
