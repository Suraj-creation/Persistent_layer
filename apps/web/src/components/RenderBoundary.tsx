import { Component, type ReactNode } from "react";

/** Keep navigation usable if a deferred bundle or view cannot render. */
export class RenderBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return this.props.fallback ?? (
      <div className="loading-state" role="alert">
        <p>This view could not load. Reload the workspace to try again.</p>
        <button className="secondary-button" onClick={() => window.location.reload()}>
          Reload workspace
        </button>
      </div>
    );
  }
}
