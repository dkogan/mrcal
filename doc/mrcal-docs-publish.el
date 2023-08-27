(setq org-publish-project-alist
      `(("orgfiles"
         :base-directory "."
         :base-extension "org"
         :publishing-directory "./out"
         :publishing-function org-html-publish-to-html
         :section-numbers nil
         :with-toc nil
         :with-sub-superscript nil
         :html-postamble nil
         :with-author "Dima Kogan"
         :with-email  "dima@secretsauce.net"
         :html-head-include-default-style nil
         :html-head ,(concat
                      "<link rel=\"stylesheet\" type=\"text/css\" href=\"org.css\"/>"
                      "<link rel=\"stylesheet\" type=\"text/css\" href=\"mrcal.css\"/>")
         :html-preamble ,(with-temp-buffer
                           (insert-file-contents "mrcal-preamble-GENERATED.html")
                           (buffer-string))
         :html-mathjax-options ((path "external/MathJax-master/es5/tex-chtml.js")
                                (scale "100")
                                (align "center")
                                (font "TeX")
                                (linebreaks "false")
                                (autonumber "AMS")
                                (indent "0em")
                                (multlinewidth "85%")
                                (tagindent ".8em")
                                (tagside "right"))
         :html-mathjax-template "<script>
  window.MathJax = {
    loader: {
      load: ['[custom]/xypic.js'],
      paths: {custom: 'external/XyJax-v3-master/build'}
    },
    tex: {
      packages: {'[+]': ['xypic']},
      ams: {
        multlineWidth: '%MULTLINEWIDTH'
      },
      tags: '%TAGS',
      tagSide: '%TAGSIDE',
      tagIndent: '%TAGINDENT'
    },
    chtml: {
      scale: %SCALE,
      displayAlign: '%ALIGN',
      displayIndent: '%INDENT'
    },
    svg: {
      scale: %SCALE,
      displayAlign: '%ALIGN',
      displayIndent: '%INDENT'
    },
    output: {
      font: '%FONT',
      displayOverflow: '%OVERFLOW'
    }
  };
</script>

<script
  id=\"MathJax-script\"
  async
  src=\"%PATH\">
</script>"

)))
