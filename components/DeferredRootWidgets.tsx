"use client";

import { useEffect, useState } from "react";
import type { ComponentType } from "react";

type WidgetProps = Record<string, never>;
type Widget = ComponentType<WidgetProps>;
type WidgetLoader = () => Promise<{ default: Widget }>;

const loadRightSidebar: WidgetLoader = () => import("./RightSidebar");
const loadPodcastPlayer: WidgetLoader = () => import("./PodcastPlayer");

function useIdleWidget(loader: WidgetLoader) {
  const [WidgetComponent, setWidgetComponent] = useState<Widget | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      loader().then(({ default: LoadedWidget }) => {
        if (!cancelled) setWidgetComponent(() => LoadedWidget);
      });
    };

    if ("requestIdleCallback" in window) {
      const idleId = window.requestIdleCallback(load, { timeout: 2000 });
      return () => {
        cancelled = true;
        window.cancelIdleCallback(idleId);
      };
    }

    const timeoutId = setTimeout(load, 1000);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [loader]);

  return WidgetComponent;
}

export function DeferredRightSidebar() {
  const Widget = useIdleWidget(loadRightSidebar);
  return Widget ? <Widget /> : null;
}

export function DeferredPodcastPlayer() {
  const Widget = useIdleWidget(loadPodcastPlayer);
  return Widget ? <Widget /> : null;
}
