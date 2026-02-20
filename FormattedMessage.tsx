import React from 'react';

interface Props {
  text: string;
}

export const FormattedMessage: React.FC<Props> = ({ text }) => {
  const renderInline = (line: string, key: number) => {
    // Process bold+italic ***text***, bold **text**, italic *text*, inline code `code`
    const parts: React.ReactNode[] = [];
    const regex = /(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`)/g;
    let last = 0;
    let match;

    while ((match = regex.exec(line)) !== null) {
      if (match.index > last) {
        parts.push(line.slice(last, match.index));
      }
      if (match[2]) {
        parts.push(<strong key={match.index}><em>{match[2]}</em></strong>);
      } else if (match[3]) {
        parts.push(<strong key={match.index}>{match[3]}</strong>);
      } else if (match[4]) {
        parts.push(<em key={match.index}>{match[4]}</em>);
      } else if (match[5]) {
        parts.push(
          <code key={match.index} style={{
            background: 'rgba(255,255,255,0.1)',
            borderRadius: '3px',
            padding: '1px 5px',
            fontFamily: 'monospace',
            fontSize: '0.875em'
          }}>{match[5]}</code>
        );
      }
      last = match.index + match[0].length;
    }

    if (last < line.length) {
      parts.push(line.slice(last));
    }

    return <React.Fragment key={key}>{parts}</React.Fragment>;
  };

  const parseBlocks = (raw: string): React.ReactNode[] => {
    const lines = raw.split('\n');
    const nodes: React.ReactNode[] = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Skip empty lines
      if (line.trim() === '') {
        i++;
        continue;
      }

      // Code block ```
      if (line.trim().startsWith('```')) {
        const lang = line.trim().slice(3).trim();
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        nodes.push(
          <pre key={i} style={{
            background: 'rgba(0,0,0,0.4)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            padding: '1rem',
            overflowX: 'auto',
            margin: '0.75rem 0',
            fontSize: '0.85rem',
            fontFamily: 'monospace',
            lineHeight: 1.6,
            color: '#e2e8f0'
          }}>
            {lang && <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.5rem' }}>{lang}</div>}
            <code>{codeLines.join('\n')}</code>
          </pre>
        );
        i++;
        continue;
      }

      // Heading # ## ###
      const headingMatch = line.match(/^(#{1,3})\s+(.+)/);
      if (headingMatch) {
        const level = headingMatch[1].length;
        const sizes = ['1.2rem', '1.05rem', '0.95rem'];
        nodes.push(
          <div key={i} style={{
            fontSize: sizes[level - 1],
            fontWeight: 700,
            color: '#e2e8f0',
            margin: '1rem 0 0.4rem',
            lineHeight: 1.4
          }}>
            {renderInline(headingMatch[2], i)}
          </div>
        );
        i++;
        continue;
      }

      // Horizontal rule ---
      if (/^[-*_]{3,}$/.test(line.trim())) {
        nodes.push(<hr key={i} style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.1)', margin: '0.75rem 0' }} />);
        i++;
        continue;
      }

      // Bullet list - or *
      if (/^[\s]*[-*•]\s+/.test(line)) {
        const listItems: React.ReactNode[] = [];
        while (i < lines.length && /^[\s]*[-*•]\s+/.test(lines[i])) {
          const itemText = lines[i].replace(/^[\s]*[-*•]\s+/, '');
          listItems.push(
            <li key={i} style={{
              marginBottom: '0.3rem',
              lineHeight: 1.65,
              color: '#e2e8f0'
            }}>
              {renderInline(itemText, i)}
            </li>
          );
          i++;
        }
        nodes.push(
          <ul key={`ul-${i}`} style={{
            paddingLeft: '1.4rem',
            margin: '0.4rem 0',
            listStyleType: 'disc'
          }}>
            {listItems}
          </ul>
        );
        continue;
      }

      // Numbered list 1. 2. etc
      if (/^[\s]*\d+\.\s+/.test(line)) {
        const listItems: React.ReactNode[] = [];
        while (i < lines.length && /^[\s]*\d+\.\s+/.test(lines[i])) {
          const itemText = lines[i].replace(/^[\s]*\d+\.\s+/, '');
          listItems.push(
            <li key={i} style={{
              marginBottom: '0.3rem',
              lineHeight: 1.65,
              color: '#e2e8f0'
            }}>
              {renderInline(itemText, i)}
            </li>
          );
          i++;
        }
        nodes.push(
          <ol key={`ol-${i}`} style={{
            paddingLeft: '1.4rem',
            margin: '0.4rem 0'
          }}>
            {listItems}
          </ol>
        );
        continue;
      }

      // Blockquote >
      if (line.startsWith('>')) {
        const quoteText = line.replace(/^>\s*/, '');
        nodes.push(
          <blockquote key={i} style={{
            borderLeft: '3px solid #22d3ee',
            paddingLeft: '0.75rem',
            margin: '0.5rem 0',
            color: '#94a3b8',
            fontStyle: 'italic'
          }}>
            {renderInline(quoteText, i)}
          </blockquote>
        );
        i++;
        continue;
      }

      // Regular paragraph
      nodes.push(
        <p key={i} style={{
          margin: '0.35rem 0',
          lineHeight: 1.7,
          color: '#e2e8f0'
        }}>
          {renderInline(line, i)}
        </p>
      );
      i++;
    }

    return nodes;
  };

  return (
    <div style={{ fontSize: '0.9375rem', wordBreak: 'break-word' }}>
      {parseBlocks(text)}
    </div>
  );
};