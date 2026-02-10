import { createBrowserRouter } from "react-router";

export const router = createBrowserRouter([
  {
    path: "/",
    lazy: async () => ({ Component: (await import("./components/Layout")).default }),
    children: [
      { index: true, lazy: async () => ({ Component: (await import("./pages/ScenarioSelector")).default }) },
      { path: "workspace", lazy: async () => ({ Component: (await import("./pages/MapWorkspace")).default }) },
      { path: "export", lazy: async () => ({ Component: (await import("./pages/ExportReport")).default }) },
    ],
  },
]);
