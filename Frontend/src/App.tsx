import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Index from "./pages/Index";
import LiveCamera from "./pages/LiveCamera";
import About from "./pages/About";
import Contact from "./pages/Contact";
import AnalyticsDashboard from "./pages/AnalyticsDashboard";
import AlertsPage from "./pages/AlertsPage";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />

      <BrowserRouter>
        <Routes>

          <Route path="/" element={<Index />} />

          <Route path="/live-camera" element={<LiveCamera />} />

          <Route path="/analytics" element={<AnalyticsDashboard />} />

          <Route path="/alerts" element={<AlertsPage />} />

          <Route path="/about" element={<About />} />

          <Route path="/contact" element={<Contact />} />

        </Routes>
      </BrowserRouter>

    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
