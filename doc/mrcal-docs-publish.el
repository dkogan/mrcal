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
         :html-head ,(concat "<style>"
                             "pre.src      {background-color: #303030; color: #e5e5e5; max-width 700px} "
                             "pre.example  {background-color: #303030; color: #e5e5e5; max-width 700px} "
                             ".org-svg     {min-width: 500px; max-width: 60%;} "
                             ".figure      {min-width: 500px; max-width: 60%;} "
                             "</style>"))
        ("images"
         :base-directory "~/jpl/mrcaldocs/"
         :base-extension "jpg\\|png\\|svg"
         :publishing-directory "./out"
         :publishing-function org-publish-attachment)

        ("website"
         :components ("orgfiles" "images"))))
