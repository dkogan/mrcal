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
         :with-author "Dima Kogan"
         :with-email  "kogan@jpl.nasa.gov"
         :html-head ,(concat "<style>"
                             "pre.src      {background-color: #303030; color: #e5e5e5; max-width 700px} "
                             "pre.example  {background-color: #303030; color: #e5e5e5; max-width 700px} "
                             ".org-svg     {min-width: 500px; max-width: 60%;} "
                             ".figure      {min-width: 500px; max-width: 60%;} "
                             "</style>")
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
        ("images"
         :base-directory "~/jpl/mrcaldocs/"
         :base-extension "jpg\\|png\\|svg"
         :publishing-directory "./out"
         :publishing-function org-publish-attachment)

        ("website"
         :components ("orgfiles" "images"))))
