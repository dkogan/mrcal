;;; Directory Local Variables

;; Useful to add links to a mrcal function or tool. These make the appropriate
;; text and an appropriate link
((org-mode . ((eval .
                    (progn
                       (defun insert-function (f)
                         (interactive (list (read-string "Function: ")))
                         (insert (format "[[file:mrcal-python-api-reference.html#-%1$s][=mrcal.%1$s()=]]"
                                         f)))

                       (defun insert-tool (f)
                         (interactive (list (read-string "Tool: ")))
                         (insert (format "[[file:%1$s.html][=%1$s=]]"
                                         f)))

                       (local-set-key (kbd "<f1>") 'insert-function)
                       (local-set-key (kbd "<f2>") 'insert-tool))

                    ))))
