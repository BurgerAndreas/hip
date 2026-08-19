# Author Checklist

**Manuscript Number:** NATCOMPUTSCI-26-0870A

Please check the items below carefully and add a response in each row of the table to indicate the changes that you have made. Please also check through any additional marked-up edits we may have provided within the manuscript file.

---

## Abstract and editor's summary

| Our guidance | Your response |
| --- | --- |
| We would like to propose a revised title to comply with our formatting requirements and improve the accessibility of your work: **HIP: Hessian Interatomic Potentials without derivatives**. Please edit the title in your manuscript files accordingly. Note that further minor changes may be made during the production process, and you will be able to check these in the proofs. | Agreed, we will use “HIP: Hessian Interatomic Potentials without derivatives” |
| Authors have the option to add a link from the published paper to the preprint version on arXiv, bioRxiv, ChemRxiv and ResearchSquare. This link will be above the abstract, visible to all readers. Please let us know if you would like to make use of this option and if so, please provide us with the link of your preprint. | https://arxiv.org/abs/2509.21624 |
| **Title:** We are only able to use colons in titles when it follows the structure "ToolName: brief title". I provided a suggestion here to comply with our production rules, but please feel free to edit further, as needed. | We will use “HIP: Hessian Interatomic Potentials without derivatives” |

---

## Author information

| Our guidance | Your response |
| --- | --- |
| We ask that you consult with your coauthors to ensure that all names, affiliations, and titles are represented correctly. Note that if any authors are added or removed after this point then all authors will be requested to provide approval documentation that could potentially delay the production of your paper. | Done |
| Please ensure that all affiliations are in the correct sequential order according to their position in the author list. Affiliation 1 must be the first affiliation associated with the first author. Please see [this article](https://www.nature.com/articles/s41467-020-16621-x.pdf) as an example. | Done |

---

## Article structure

| Our guidance | Your response |
| --- | --- |
| We can accommodate up to 6 display items (figures, tables, or boxes) in the main article and up to 10 Extended Data items (figures, tables, or boxes), which will be integrated into the full-text HTML version of your paper and will be appended to the online PDF. Each Extended Data item must be cited in order in the main text. Each display item and Extended data item must fit easily within an A4 page (210 × 297 mm). Please ensure that the number and size of your display and Extended Data items fulfil these requirements to avoid any delay in the acceptance of your article. | 5 figures, 1 table |
| Please ensure that your revised manuscript does not exceed 3500 words. | Done. Introduction, Results and Discussion are approximately 2,700 words (counted from `main.tex` after stripping comments, citations and display equations), below the 3,500-word limit. Methods, captions and references are excluded. |
| Please ensure your main manuscript file includes the following sections, in this order: Title; Author list; Affiliations; Abstract; Main text; Methods; Data Availability; Code Availability (if relevant); Acknowledgements; Funding Statement; Author Contributions Statement; Competing Interests Statement; Tables; Figure Legends/Captions (for main text figures); References. | Done through Competing Interests. Following the production note to place display-item captions after the reference list, Table 1 and the figure legends follow the inlined references. |
| Please also divide your Main text section into the following sections: Introduction / Results (with at least one subheading) / Discussion. | Done |
| Please supply source data for the following figures: Figures 2–5. Source data for these figures should be the raw numerical data behind the plots (not the images of the plots). If the plots depict mean/median/box plots, please supply all of the data points used to compute the statistics. Please upload one file per main figure; you can use `.zip` if necessary. **Notes on Source Data format:** Numerical source data files may be supplied in `.xls(x)`, `.csv`, `.txt` or `.zip` format. Image source data files may be supplied in `.tif`, `.jpg`, `.pdf` or `.zip` format. Separate files should be supplied for each Figure. Source Data files must be listed in the Inventory of Supporting Information. Source Data files may be provided for Figures and Extended Data only. For Supplementary Information items, please supply source data as Supplementary Data files (or for image data such as uncropped, unprocessed scans of blots and gels, at the end of the Supplementary Information file). | We provide separate ZIP archives for Figures 2–5 containing the numerical source data used to calculate the reported statistics of each panel. Numerical data for Supplementary Figures and Tables are in `Supplementary_Data.zip`. All files are listed in the Inventory of Supporting Information. |

---

## Main text

| Our guidance | Your response |
| --- | --- |
| **General** | |
| 1. If submitting the main article file as LaTeX, please make sure that the LaTeX file is submitted as a *single*, complete file rather than separate bibliography or style files. For instance, you can copy the reference list from the `.bbl` file, paste it into the main manuscript `.tex` file, and delete the associated `\bibliography` and `\bibliographystyle` commands. Before submission, please ensure that the complete `.tex` file compiles successfully on your own system with no errors. | The reference list is inlined in `main.tex` and `\bibliography` has been removed. |
| 2. Please refer to sections/notes in the Supplementary Information as “Supplementary Section #” and Supplementary Figures and Tables as “Supplementary Figure #” and/or “Supplementary Table #” both in the main text and in the SI. | The main text does not cite the SI (the former batching pointer is commented out). In `si.tex`, captions use `Supplementary Figure`/`Supplementary Table`, and in-text citations use those names with `\ref`. SI section headings are unnumbered titles because the main text does not point to Supplementary Sections. |
| 3. Please only use significant(ly) for instances of statistical significance. Otherwise, please use substantial(ly). | Done. We replaced non-statistical uses of significant(ly) with substantial(ly) in the main text. |
| 4. Please change instances of e.g. to “for example” or “for instance” and instances of i.e. to “that is” or “meaning”. | No instances of “e.g.” or “i.e.” remain. |
| 5. Please avoid the use of overhyped language such as new/novel/first. | Non-technical uses of “new”, “novel” and “first” were removed; technically necessary terms such as “first-order saddle” were retained. |
| 6. Please ensure that headings and sub-headings are not longer than 60 characters including spaces and that they do not use active verbs. | Headings were checked and revised to use nominal, sentence-case phrasing; all are under 60 characters. |
| 7. When referring readers to the Methods section or the SI, please provide a specific section for them to refer to. | Done. Main-text Methods pointers name the relevant subsection (for example, “Methods, Glycine proton-transfer case study”). Within Methods, cross-references also name the target subsection. |
| 8. Please be sure to define all variables and acronyms throughout the manuscript. | Remaining acronyms, including MLIP, AD, HVP, TS, PES, HORM and ZPE, were defined at first use. Grammar and number agreement were also corrected. |
| 9. Please remove keywords from lines 36/37. | Done. The `\keywords` line has been removed. |
| **Results** | |
| 1. In this section, please only provide methodological details that are necessary for the readers' understanding of the results. All other, more technical details, should be reserved for the Methods section. | Done. Technical details on grid construction, software, batching, benchmark composition, and training budgets have been moved to Methods. |
| 2. Line 115: Please provide references for m ωB97X and 6-31G*. | Done. We added the original references for the $\omega$B97X functional and the 6-31G* basis set. |
| **Discussion** | |
| Please note that we do not use conclusions sections. Instead, please provide a Discussion section that (1) does not have summaries of the methods/results presented in the paper and does not introduce new results, (2) has a discussion on the *limitations* of the approach (this is very important!), and (3) has a discussion of which experiments could be performed in the future to better validate the results and expand the usability. Please revise this section accordingly. All relevant discussions of the data should be provided in the Results section. This section should not have sub-sections. | Done. We replaced the conclusion-style summary with a single Discussion section without sub-sections or new results. It addresses integrability and energy--force--Hessian consistency, training-data coverage, finite-cutoff effects and the cost of reference Hessians. It also proposes validation across broader chemical spaces and higher-fidelity electronic-structure methods, together with extensions to periodic systems and higher-rank response tensors. |
| **Supporting Information** | |
| 1. Please note that we do not use appendices. Instead, please provide a separate Supplementary Information file and remove the appendix from the main text file. | Done. The appendix has been removed from `main.tex` and placed in a separate `si.tex`. |
| 2. Please note that Figures 2–5 are mixed in with the Appendix. In your revision, please upload the main text figures individually (removing them from the article file) and keep the corresponding captions at the end of the file, after the references. Table 1 should also be moved to the end of the article file, along with the corresponding caption. | Done. `\includegraphics` has been removed from `main.tex`. Table 1 and caption-only figure legends are placed after the reference list. Figures are supplied as separate vector PDFs: `Figure1.pdf`--`Figure5.pdf`. |
| 3. Please note that we do not allow supplementary methods. As such, please move all methodological details to the main text methods section, keeping any associated figures/tables in the SI. | Done. Experimental and computational details have been moved to the main Methods section; associated supporting figures and tables remain in the Supplementary Information. |
| 4. Please ensure that all supplementary figures/tables have complete captions that include both a title and a brief description of what is depicted. | Done. Each remaining SI figure and table has a noun-phrase title, a short description of what is shown, defined encodings, and a precise \(n\) where statistics are reported. |

---

## Figures and Tables

See the [guidelines for preparing final artwork](https://www.nature.com/documents/NRJs-guide-to-preparing-final-artwork.pdf). Following these instructions will reduce the chances of delays should we need to request replacement artwork from you at a later stage.

- Tables must be editable and prepared using the table menu in Word, the table environment of LaTeX, or the table functionality of Chem Draw where relevant.
- If bold/italic formatting in tables is necessary, please define its meaning in a table footnote.
- Shadings or symbols in graphs must be defined in some fashion. We prefer that you use a key within the image; do not include coloured symbols in the legend/caption.

| Our guidance | Your response |
| --- | --- |
| **General** | |
| 1. Please be sure that all variables and acronyms are defined in the corresponding caption. | Done. We defined the variables, model abbreviations, computational methods and statistical quantities used in each caption. |
| 2. Please be sure to explain all colors/shadings in the captions. | Done. We explained the colour, line-style, marker and distribution-shading encodings in the corresponding captions. |
| 3. Note that captions cannot be more than 250 words long. | Done. Each figure caption is fewer than 250 words. |
| 4. Please clarify if there is any third-party content in any of your figures. | All figures were created by the authors and contain no third-party content. |
| 5. Please ensure that all figures have complete captions that include both a title and a brief description of what is depicted. Note that figure titles cannot use active verbs. | Done. Each caption now begins with a noun-phrase title and briefly describes every panel. |
| 6. Wherever molecules are depicted, please provide a key indicating atom color/character. | Done. The Figure 1 caption clarifies that its nodes are generic atoms rather than chemical elements, and the Figure 2 caption provides the atom-colour key. |
| 7. Wherever hydrogen atoms are omitted, please indicate that's the case in the caption. | Done. Figure 2 states that all hydrogen atoms are shown; Figure 1 uses generic graph nodes rather than an element-specific molecular rendering. |
| **Figure 2** | |
| 1. Please note that the colors used for 2 and 3 in part A are quite similar and are difficult to discern. Please update the color palette to improve readability. | We replaced the palette in panel a with the perceptually uniform `magma` palette, which provides stronger separation between adjacent integer mode counts, particularly counts 2 and 3. |
| 2. Please note that the two greys used in part C are difficult to discern from one another. Please update the color palette or add hash marks/dashes to improve readability. | We changed the LeftNet-CF AD curves to solid purple and EqV2 AD to dotted blue, making the two methods distinguishable by both colour and line style. |
| **Figure 3** | |
| 1. Same comment on colors as Figure 2. | We revised the palette in panels a and b: direct-force AD is blue, conservative-force AD purple, finite differences dark grey, the forward pass teal and HIP orange. |
| **Figure 4** | |
| 1. Please add error bars. | We added 95% bootstrap confidence intervals to the ZPE errors in panel c and the stationary-point classification accuracies in panel d. Panel a shows full sample distributions and panel b reports stagewise counts, for which error bars are not applicable. |
| **Figure 5** | |
| 1. Sorry, I am blue/yellow colorblind, so it's very difficult for me to differentiate the blues/greens in parts d–f. Please update the color palette, if possible, to improve readability. | We replaced the original blue–green scale in panels d–f with a purple–red–yellow palette, providing stronger hue and lightness separation. |

---

## Data and Code

Nature journals strongly support public availability of data and code. Please deposit the data and code used in your paper into a public data repository, or alternatively, present the data as Supplementary Information. If data can only be shared on request, please explain why in the Data Availability Statement, and also in the correspondence with your editor.

Please note that for some data types, deposition in a public repository is mandatory. Any restrictions on sharing of these data types must be clearly indicated in the statement and discussed with the editor. More information: [availability of data](https://www.nature.com/nature-research/editorial-policies/reporting-standards#availability-of-data).

All published manuscripts reporting original research in Nature Portfolio journals must include a data availability statement, within the Methods and under the heading “Data Availability”.

The data availability statement must make the conditions of access to the “minimum dataset” that are necessary to interpret, verify and extend the research in the article, transparent to readers.

This minimum dataset may be provided through deposition in public community/discipline-specific repositories, custom proprietary repositories or general repositories like Figshare, Zenodo and Dryad. Providing large datasets in supplementary information is strongly discouraged and the preferred approach is to make data available in repositories. [Approved and recommended data repositories](https://www.nature.com/sdata/policies/repositories).

The Data Availability Statement should also reference any source data published alongside the paper.

If DOIs are provided, we also require including these in the Reference list (authors, title, publisher (repository name), identifier, year).

For clinical datasets or third party data, please ensure that the statement adheres to the [policy](https://www.nature.com/nature-research/editorial-policies/reporting-standards#availability-of-data).

When electronic structure calculations are reported in the manuscript, the atomic coordinates of the optimized computational models should be provided. For molecular dynamics trajectories at least the initial and final configurations should be supplied. We encourage you to make them available by uploading the structures in any of the existing data repositories. Alternatively, they can be supplied as a separate Supplementary Data file (ideally as a plain, unformatted text file).

| Our guidance | Your response |
| --- | --- |
| **Data Availability Statement** | See the Data Availability section in `main.tex`. |
| 1. Please indicate “Source data for Figures X is available with this manuscript”, where X represents the indices of the figures for which source data is made available. | Source data for Figures 2–5 is available with this manuscript. |
| 2. The datasets that you used should be made available through a DOI-minting repository, which can then be added to the reference list and cited from the Data Availability Statement. We recommend using Zenodo or Figshare, if needed. Please be sure to also provide all accession codes where necessary. | Training and validation labels are from HORM, archived at Zenodo (`https://doi.org/10.5281/zenodo.17217897`). Transition1x is available at figshare (`https://doi.org/10.6084/m9.figshare.19614657.v4`). DFT calculations generated in this work and trained model weights are archived at Zenodo (`https://doi.org/10.5281/zenodo.22003643`) and mirrored at Hugging Face (`https://huggingface.co/andreasburger/hip`). These DOIs are cited from the Data Availability Statement and the reference list. |
| **Code Availability Statement** | The HIP code is available at `https://github.com/BurgerAndreas/hip` under the Apache License, Version 2.0. A frozen snapshot is archived at Zenodo (`https://doi.org/10.5281/zenodo.22003592`) and cited from the Code Availability Statement and the reference list. |

---

## Methods

| Our guidance | Your response |
| --- | --- |
| 1. Please be sure to define all variables. | Done. Methods now define \(N\), \(l\), \(m\), \(c\), \(T\), Clebsch--Gordan coefficients, the subspace weight \(\alpha\) and index \(k\), atomic number \(Z\), per-molecule MAE \(x_i\), and E-F (energy-and-force-only training). |
| 2. Please note that the manuscript must be self-contained. As such, please avoid referring readers to other manuscripts for methodological details and provide all necessary details here. | Added details on the architecture, optimizer, SCF thresholds, ReactBench success criteria, and all DFT protocols to the Method section. |
| 3. Apologies if I missed it, but please provide DFT methodology here, including the software and version number. | Methods now report GPU4PySCF v1.3.0 (\(\omega\)B97X/6-31G*) for HORM labels, HORM-RGD1, and ReactBench verification; ORCA 6.0 (\(\omega\)B97X/6-31G(d)) for PubChem; and ORCA 6.1.1 (\(\omega\)B97X-D3/6-31G(d)) for glycine, including calculation types (Opt, EnGrad, Freq). |

---

## References

| Our guidance | Your response |
| --- | --- |
| 1. Please check any references that point to preprints. If they have since been published, please update the reference accordingly. | We checked all cited preprint entries. Four have since been published and were updated: Fu et al. (ICML 2025), Bigi et al. (ICML 2025), Li et al. (ICLR 2025), and Eberhard et al. (ICML 2026). Three remain preprints with no journal or proceedings version: Thomas et al., arXiv:1802.08219; Duval et al., arXiv:2312.07511; Geiger and Smidt, arXiv:2207.09453. |
| Please ensure your LaTeX file is submitted as a single, complete file rather than separate bibliography or style files. If you wish to use BibTeX, please copy the reference list from the `.bbl` file, paste it into the main manuscript `.tex` file, and delete the associated `\bibliography` and `\bibliographystyle` commands. Before submission, please ensure that the complete `.tex` file compiles successfully on your own system with no errors or warnings. | The reference list from `main.bbl` is now inlined in `main.tex`, and the `\bibliography` command has been removed. |

---

## End matter

| Our guidance | Your response |
| --- | --- |
| Please supply an “Author Contributions” section after the “Acknowledgements” section that refers to all authors. For more information, see the [authorship policy](https://www.nature.com/nature-research/editorial-policies/authorship) and this [Nature Editorial](https://www.nature.com/articles/4581078a). | The Author Contributions section was revised to use consistent, disambiguated initials and explicitly name all eight authors. |
| Nature Portfolio defines Competing Interest (CI) as financial and non-financial interests (including but not limited to funding, employment, stocks, shares, patents, personal or professional relationships with individuals or institutions, and unpaid membership advocacy) that could be perceived to directly undermine the objectivity, integrity, and value of a publication, or could be seen as having an influence on the judgments and actions of authors with regard to objective data presentation, analysis, and interpretation. Please thoroughly review our [policy on Competing Interests](https://www.nature.com/nature-research/editorial-policies/competing-interests) and include a detailed statement both in your final manuscript file and in our manuscript tracking system. Please ensure the statements are identical in both. Be specific about how each point stated relates to the research and list applicable author initials, and/or patent numbers. If there are no competing interests, a negative statement must be included. | The manuscript includes the negative statement “The authors declare no competing interests.” The same wording will be entered in the manuscript tracking system. |
| Any relevant funding should be declared in a separate funding statement. Please refer to the [funding statement guidelines](https://www.nature.com/nature-portfolio/editorial-policies/funding) for more information. | A separate Funding section follows Acknowledgements. Initials match the Author Contributions disambiguation (`A.Bu.` for Andreas Burger). |

---

## Additional Revisions

| Our guidance | Your response |
| --- | --- |
| For any Supplementary Figures, please check and confirm that: | |
| If data is presented as bar charts, individual data points are shown using overlaid dot plots. | No SI bar charts remain. The loss-ablation relaxation figure uses violins with overlaid individual geometries. |
| The n number (that is, the sample size used to derive statistics) is provided and defined as a precise value (not a range), using the wording “n=X samples/cells/independent experiments” etc. where applicable. | Done. Captions state \(n=10\) timing repeats, \(n=80\) geometries, \(n=579\) glycine points, \(n=960\) reactions, \(n=1000\) validation geometries, and \(n=47\) ZPE pairs. |
| Any chart axis, error bars, scale bars, molecular weight markers, symbols and colour scales are defined. | Done. SI figure captions define axes, markers, line styles, colour scales and stationary-point symbols. The batching figure includes an in-image legend. |
| Any statistical tests used for data analysis are specified and exact p-values are provided either on the figures themselves, in the legend or in the Source Data file. | No inferential statistical tests were used for the SI figures or tables. Captions state this. Standard deviations in the ZPE table are sample standard deviations of the signed reactant-ZPE and $\Delta$ZPE errors. |

---

## Preparing your manuscript files

| Our guidance | Your response |
| --- | --- |
| Unless otherwise stated please limit individual file sizes to approximately 30 MB. We strongly encourage the use of repositories for large datasets or source data due to size considerations. | Done. Individual figure PDFs are under 1 MB. The largest source-data archive is `Figure_3.zip` (~3.8 MB). Larger datasets are in Zenodo/figshare repositories. |
| Please supply a brief (maximum 250 characters, including spaces) summary of the main findings of the paper to be used on our website and in our e-alerts. The summary should be written in the third person in language suitable for a broad audience. The summary may be edited by the editors prior to publication. Please provide this summary in your cover letter. | Researchers report HIP, a machine-learning model that predicts molecular Hessians directly rather than by differentiation. HIP is faster and more accurate for geometry optimization, transition-state search and vibrational analysis. (231 characters including spaces.) This text will also appear in the cover letter. |
| To ensure maximum visibility for your work, we may post about your paper following publication. If you would like us to include the X (formerly Twitter) and/or Bluesky handles of the first author(s), corresponding author(s), lab or institution in this post, please provide them in your cover letter. We would also welcome your suggestions for hashtags to use when posting about the work. | The cover letter lists the first author's X (`@atAndreasBurger`) and Bluesky (`@andreasburger.bsky.social`) handles, the Matter Lab's Bluesky handle (`@thematterlab.bsky.social`), and the suggested hashtags `#ComputationalChemistry`, `#MachineLearning` and `#AIforScience`. |
| Please supply the main manuscript file in either Microsoft Word or LaTeX format. | The main article is supplied as a single LaTeX file, `main.tex`. |
| Please provide figures as individual vector files with editable text. Acceptable file types for figures are `.ai`, `.eps`, `.pdf`, `.ppt` or Chem Draw for fully editable vector-based art. For detailed guidance, see the [artwork guidelines](https://www.nature.com/documents/aj-artworkguidelines.pdf). | Done. Main-text figures are individual vector PDFs with editable text: `Figure1.pdf`--`Figure5.pdf`. |
| Please supply the main Supplementary Information file as a single PDF file. | The Supplementary Information will be compiled from `si.tex` as a single PDF for upload. |
| It is your responsibility to obtain the right to use any items (figures, tables, images, movies or text boxes) that are reproduced (or adapted) from material for which you do not hold copyright and to give proper attribution to the creators of that work. This includes work that has previously been published elsewhere. If you do not hold the copyright for any item (in whole or part), included in your paper, you must complete and return a [Third Party Rights Table](https://www.nature.com/documents/thirdpartyrights-origres.docx). If any elements of your submitted work have been created with BioRender you will need to ensure you have obtained a publication license from BioRender, adhering to the user requirements as outlined within the license. The reference for BioRender created graphics should be present in the accompanying legend of the display material it is present in. A copy of the publication license should be uploaded to our system as a related manuscript file upon resubmission. [BioRender knowledge article](https://help.biorender.com/hc/en-gb/articles/21283116932765-CC-BY-publishing-and-reader-permissions). For more information on what constitutes ownership by a third party, please contact our Editorial Assistant at computationalscience@nature.com. | Not applicable. All figures were created by the authors. None were reproduced or adapted from third-party or BioRender material, so a Third Party Rights Table is not required. |

---

## Forms to complete

**Inventory of Supporting Information**

The inventory must be completed with details of all Supplementary Information, Extended Data and Source Data files. Download from: [Inventory of Supporting Information](http://www.nature.com/documents/Inventory_of_Supporting_Information_2021.docx).

### Files to upload

- Completed Third Party Rights Table (if relevant)
- A point-by-point response to the reviewers' comments
- A completed copy of this checklist in `.docx` format
- The main article file in LaTeX format
- Separate Figure files (one file per figure)
- Separate Source Data files
- Inventory of Supporting Information in `.docx` format
- A Supplementary Information file in `.pdf` format

---

## Questions and TODO

- Word count: Introduction–Discussion is approximately 2,700 words (under 3,500). Methods are additional and are not counted toward that limit.
- SI cross-references: the main text does not cite the SI. The SI uses “Supplementary Figure/Table” naming in captions and running text.
