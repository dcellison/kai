import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

interface ConfirmationRequest {
  message: string;
  resolve: (confirmed: boolean) => void;
}

const ConfirmationContext = createContext<((message: string) => Promise<boolean>) | null>(null);

export function ConfirmationProvider({ children }: { children: ReactNode }): React.JSX.Element {
  const [request, setRequest] = useState<ConfirmationRequest | null>(null);
  const requestRef = useRef<ConfirmationRequest | null>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);

  const settle = useCallback((confirmed: boolean): void => {
    const current = requestRef.current;
    if (!current) return;
    requestRef.current = null;
    setRequest(null);
    current.resolve(confirmed);
  }, []);

  const confirm = useCallback((message: string): Promise<boolean> => {
    requestRef.current?.resolve(false);
    return new Promise<boolean>((resolve) => {
      const next = { message, resolve };
      requestRef.current = next;
      setRequest(next);
    });
  }, []);

  useEffect(() => () => {
    requestRef.current?.resolve(false);
    requestRef.current = null;
  }, []);

  useEffect(() => {
    if (!request) return;
    cancelRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.preventDefault();
        settle(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [request, settle]);

  return (
    <ConfirmationContext.Provider value={confirm}>
      {children}
      {request && (
        <div
          className="confirmation-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) settle(false);
          }}
        >
          <section
            className="confirmation-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirmation-dialog-title"
            aria-describedby="confirmation-dialog-message"
          >
            <p className="overline">Confirm action</p>
            <h2 id="confirmation-dialog-title">Continue?</h2>
            <p id="confirmation-dialog-message">{request.message}</p>
            <div>
              <button
                ref={cancelRef}
                type="button"
                className="quiet-button"
                onClick={() => settle(false)}
              >
                Cancel
              </button>
              <button type="button" className="primary-button" onClick={() => settle(true)}>
                Continue
              </button>
            </div>
          </section>
        </div>
      )}
    </ConfirmationContext.Provider>
  );
}

export function useConfirmation(): (message: string) => Promise<boolean> {
  const confirm = useContext(ConfirmationContext);
  if (!confirm) {
    throw new Error("useConfirmation must be used within ConfirmationProvider");
  }
  return confirm;
}
