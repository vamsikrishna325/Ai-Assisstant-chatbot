import React, { useState } from 'react';

interface Props { text: string; }

// ── Inline parser: **bold**, *italic*, `code`, _italic_ ──────────────────────
const inline = (line: string, key?: number): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  const rx = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|_(.+?)_/g;
  let last = 0, m: RegExpExecArray | null;
  while ((m = rx.exec(line)) !== null) {
    if (m.index > last) parts.push(line.slice(last, m.index));
    if (m[1]) parts.push(<strong key={m.index}>{m[1]}</strong>);
    else if (m[2]) parts.push(<em key={m.index}>{m[2]}</em>);
    else if (m[3]) parts.push(<code key={m.index} style={S.inlineCode}>{m[3]}</code>);
    else if (m[4]) parts.push(<em key={m.index}>{m[4]}</em>);
    last = m.index + m[0].length;
  }
  if (last < line.length) parts.push(line.slice(last));
  return <React.Fragment key={key}>{parts.length ? parts : line}</React.Fragment>;
};

// ── Copy button for code blocks ───────────────────────────────────────────────
const CopyBtn: React.FC<{ code: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button onClick={copy} style={S.copyBtn} title="Copy code">
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  );
};

// ── Main component ────────────────────────────────────────────────────────────
export const FormattedMessage: React.FC<Props> = ({ text }) => {
  const lines = text.split('\n');
  const out: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const l = lines[i];

    // H1
    if (/^#\s/.test(l)) {
      out.push(<h1 key={i} style={S.h1}>{inline(l.replace(/^#+\s/, ''))}</h1>);
    }
    // H2
    else if (/^##\s/.test(l)) {
      out.push(
        <h2 key={i} style={S.h2}>
          <span style={S.diamond} />
          {inline(l.replace(/^#+\s/, ''))}
        </h2>
      );
    }
    // H3
    else if (/^###\s/.test(l)) {
      out.push(<h3 key={i} style={S.h3}>{inline(l.replace(/^#+\s/, ''))}</h3>);
    }
    // HR
    else if (/^(---|___|\*\*\*)$/.test(l.trim())) {
      out.push(<hr key={i} style={S.hr} />);
    }
    // Numbered list — collect consecutive items
    else if (/^\d+\.\s/.test(l)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i]))
        items.push(<li key={i} style={S.li}>{inline(lines[i].replace(/^\d+\.\s/, ''), i++)}</li>);
      out.push(<ol key={`ol${i}`} style={S.ol}>{items}</ol>);
      continue;
    }
    // Bullet list — collect consecutive items
    else if (/^[-*]\s/.test(l)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^[-*]\s/.test(lines[i]))
        items.push(<li key={i} style={S.li}>{inline(lines[i].replace(/^[-*]\s/, ''), i++)}</li>);
      out.push(<ul key={`ul${i}`} style={S.ul}>{items}</ul>);
      continue;
    }
    // Code block
    else if (/^```/.test(l)) {
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) codeLines.push(lines[i++]);
      i++; // skip closing ```
      const code = codeLines.join('\n');
      out.push(
        <div key={`cb${i}`} style={S.codeWrap}>
          <CopyBtn code={code} />
          <pre style={S.pre}>{code}</pre>
        </div>
      );
      continue;
    }
    // Label line e.g. "Handles:"
    else if (/^[A-Za-z][A-Za-z\s]+:$/.test(l.trim())) {
      out.push(<p key={i} style={S.label}>{inline(l)}</p>);
    }
    // Empty line
    else if (!l.trim()) {
      out.push(<div key={i} style={{ height: '0.3rem' }} />);
    }
    // Plain text
    else {
      out.push(<p key={i} style={S.p}>{inline(l)}</p>);
    }

    i++;
  }

  return <div style={{ display: 'flex', flexDirection: 'column', gap: '0.05rem' }}>{out}</div>;
};

// ── Styles ────────────────────────────────────────────────────────────────────
const S: Record<string, React.CSSProperties> = {
  h1:         { fontSize: '1.3rem',  fontWeight: 700, color: '#e2e8f0', margin: '0.9rem 0 0.3rem' },
  h2:         { fontSize: '1.05rem', fontWeight: 700, color: '#e2e8f0', margin: '0.75rem 0 0.2rem', display: 'flex', alignItems: 'center', gap: '0.4rem' },
  h3:         { fontSize: '0.96rem', fontWeight: 600, color: '#cbd5e1', margin: '0.6rem 0 0.15rem' },
  diamond:    { width: 6, height: 6, background: '#60a5fa', borderRadius: 2, display: 'inline-block', flexShrink: 0 },
  hr:         { border: 'none', borderTop: '1px solid rgba(255,255,255,0.1)', margin: '0.65rem 0' },
  ol:         { paddingLeft: '1.4rem', margin: '0.25rem 0', listStyleType: 'decimal' },
  ul:         { paddingLeft: '1.3rem', margin: '0.25rem 0', listStyleType: 'disc' },
  li:         { fontSize: '0.92rem', color: '#cbd5e1', marginBottom: '0.18rem' },
  p:          { fontSize: '0.92rem', color: '#cbd5e1', margin: '0.12rem 0', lineHeight: 1.6 },
  label:      { fontSize: '0.9rem',  color: '#94a3b8', margin: '0.35rem 0 0.1rem', fontWeight: 500 },
  inlineCode: { background: 'rgba(255,255,255,0.1)', borderRadius: 4, padding: '1px 5px', fontFamily: 'monospace', fontSize: '0.87em' },
  codeWrap:   { position: 'relative', margin: '0.45rem 0' },
  pre:        { background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 6, padding: '0.65rem 1rem', fontSize: '0.84rem', fontFamily: 'monospace', color: '#a5f3fc', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 },
  copyBtn:    { position: 'absolute', top: 6, right: 8, background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 4, color: '#94a3b8', fontSize: '0.75rem', padding: '2px 8px', cursor: 'pointer', zIndex: 1 },
};

export default FormattedMessage;
