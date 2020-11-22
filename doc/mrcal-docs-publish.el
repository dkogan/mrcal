;; Python API and the cmdline tool manpages
(defun dima-org-export-build-docs (plist)
  (shell-command (format "make -C .. doc")))


;; C API
(defun dima-org-export-htmlized-header (plist filename pub-dir)
  (htmlize-file filename pub-dir))

(setq org-publish-project-alist
      '(("orgfiles"
         :base-directory "."
         :base-extension "org"
         :publishing-directory "./out"
         :publishing-function org-html-publish-to-html
         :completion-function dima-org-export-build-docs
         :section-numbers nil
         :with-toc t
         :html-preamble nil)

        ("images"
         :base-directory "~/jpl/mrcaldocs/"
         :base-extension "jpg\\|png\\|svg"
         :publishing-directory "./out"
         :publishing-function org-publish-attachment)

        ;; marked-up public C headers for documentation
        ("headers"
         :base-directory ".."
         :include ("mrcal.h" "poseutils.h" "basic_geometry.h")
         :publishing-directory "./out"
         :publishing-function dima-org-export-htmlized-header)

        ("website"
         :components ("orgfiles" "images" "headers"))))
