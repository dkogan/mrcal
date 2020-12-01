;; Python API and the cmdline tool manpages
(defun dima-org-export-build-docs (plist)
  (shell-command (format "make -C .. doc")))

(setq org-publish-project-alist
      `(("orgfiles"
         :base-directory "."
         :base-extension "org"
         :publishing-directory "./out"
         :publishing-function org-html-publish-to-html
         :completion-function dima-org-export-build-docs
         :section-numbers nil
         :with-toc t
         :with-sub-superscript nil
         :html-postamble nil
         :with-author "Dima Kogan"
         :with-email  "kogan@jpl.nasa.gov"
         :html-head-include-default-style nil
         :html-head ,(concat
                      "<link rel=\"stylesheet\" type=\"text/css\" href=\"org.css\"/>"
                      "<link rel=\"stylesheet\" type=\"text/css\" href=\"mrcal.css\"/>")
         :html-preamble ,(with-temp-buffer
                           (insert-file-contents "mrcal-preamble.html")
                           (buffer-string))
         :html-mathjax-options ((path "MathJax-master/es5/tex-chtml.js")
                                (scale "100")
                                (align "center")
                                (font "TeX")
                                (linebreaks "false")
                                (autonumber "AMS")
                                (indent "0em")
                                (multlinewidth "85%")
                                (tagindent ".8em")
                                (tagside "right")))
        ("css"
         :base-directory "."
         :base-extension "css"
         :publishing-directory "./out"
         :publishing-function org-publish-attachment)

        ("images"
         :base-directory "~/jpl/mrcaldocs/"
         :base-extension "jpg\\|png\\|svg"
         :publishing-directory "./out"
         :publishing-function org-publish-attachment)

        ("website"
         :components ("orgfiles" "css" "images"
                      ))))
