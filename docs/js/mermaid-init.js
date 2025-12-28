document.addEventListener('DOMContentLoaded', function() {
  mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true,
      curve: 'basis'
    }
  });

  // Convert pre.mermaid > code elements for mermaid rendering
  document.querySelectorAll('pre.mermaid').forEach(function(pre) {
    var code = pre.querySelector('code');
    if (code) {
      pre.textContent = code.textContent;
    }
  });

  mermaid.run();
});
