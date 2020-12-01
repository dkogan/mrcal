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
         :html-head ,(concat "<style>"
                             "body         {background-color: burlywood;} "
                             "pre          {border: #303030; box-shadow: 3px 3px 3px #606060;} "
                             "pre.src      {background-color: #303030; color: #e5e5e5; max-width 700px} "
                             "pre.example  {background-color: #303030; color: #e5e5e5; max-width 700px} "

                             ".org-svg     {width: 90%; min-width: 500px; max-width: 900px;} "

                             ;;  Style for the nav bar at the top. Adapted from
                             ;;  o-blog.css
                             ".supernav_title { "
                             "    font-size: 30px; "
                             "    font-weight: bolder; "
                             "} "
                             ".supernav_list { "
                             "    vertical-align: middle; "
                             "    top: 0; "
                             "    list-style: none outside none; "
                             "    white-space: nowrap; "
                             "    overflow: hidden; "
                             "} "
                             ".supernav_list li { "
                             "    display: inline-block; "
                             "} "
                             ".supernav_list li + li:before { "
                             "    content: \" / \"; "
                             "    padding: 0 10px; "
                             "} "

                             "</style>")

         :html-preamble ,(concat
                          "<ul class=\"supernav_list\"> "
                          "<li class=\"supernav_title\">mrcal</li> "
                          "<li><a href=\"index.html\">Documentation index</a></li> "
                          "<li><a href=\"tour.html\">A tour of mrcal</a></li> "
                          "<li><a href=\"c-api.html\">C API</a></li> "
                          "<li><a href=\"python-api.html\">Python API</a></li> "
                          "<li><a href=\"commandline-tools.html\">Commandline tools</a></li> "
                          "<li><a href=\"https://github.jpl.nasa.gov/maritime-robotics/mrcal\">Sources</a></li> "
                          "</ul> <hr>")

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
