# What to say: the model part

About 190 words with both slots filled: close to 90 seconds at a calm pace (130 words a minute).
Leaving out the parts marked (can cut) brings it to about 80 seconds. Each English line is one
breath. The Chinese line under it says the same thing.

Numbers are said as "42 of 44", never as a percentage.

## Script

**Slide: the module diagram**

1. Our part is the model. For each corn field it answers one question: silage or grain?
   > 我们这部分是模型。它对每一块玉米地回答一个问题：青贮还是籽粒？

2. First it checks the evidence. Did the satellite see the harvest?
   > 第一步是检查证据：卫星有没有拍到收割？

3. If clouds hid the harvest, we do not guess. The field is "not judged".
   > 如果云挡住了收割，我们不猜。这块地记为"不判断"。

4. Then a small model decides. It uses six numbers.
   > 然后由一个小模型做判断。它只用六个数。

5. The key one is moisture just before harvest. Silage is cut wet. Grain is left to dry.
   > 最关键的一个是收割前的含水量。青贮是趁湿割的，籽粒玉米要等它变干。

**Slide: the labelled fields (fig3)**

6. We trained only on European fields. The model never saw a US field.
   > 我们只用欧洲的地块训练。模型从没见过美国的地。

7. Then we tested it on the organisers' labelled fields. It got 42 of 44 right.
   > 然后我们在主办方带标签的地块上测试。44 个里对了 42 个。

8. A much bigger model got 37. Always saying "grain" gets 32. (first sentence can cut)
   > 一个大得多的模型对了 37 个。全部猜"籽粒"能对 32 个。（第一句可删）

**Slide: counties and the state (fig5, state_2025)**

9. Last, we add the fields up to county and state acres.
   > 最后，我们把地块加起来，得到县和州的面积。

10. Here we must be honest. Our first county method failed its test in Maryland.
    > 这里必须说实话：我们第一种县级方法在马里兰没有通过检验。

11. One cause is cloud at harvest time. Now we count silage only where we saw the harvest, and we report the corn we could not see.
    > 原因之一是收割期有云。现在我们只在看到了收割的地方才计青贮，并且报告没看到的玉米有多少。

12. **SLOT A, Michigan (to come).** Say the one that matches the logged result:
    - passed: "On two new counties in Michigan, this method passed its test."
      > 在密歇根两个新的县上，这个方法通过了检验。
    - failed: "On two new counties in Michigan it failed again. So today our county numbers are only a rough guide."
      > 在密歇根两个新的县上它又没通过。所以目前我们的县级数字只能作粗略参考。
    - not testable: "In Michigan the clouds hid too much of the harvest, so we could not test it there."
      > 在密歇根，云挡住了太多收割期，所以没法在那里检验。

13. **SLOT B, Maryland 2025 (to come).** Say the one that matches:
    - a number is reported: "For Maryland in 2025 we estimate ______ acres of silage. USDA reports 40,000."
      > 对 2025 年的马里兰，我们估计青贮面积是 ______ 英亩。美国农业部的数字是 4 万。
    - no number is reported: "For Maryland in 2025 the images do not support a state number, and we say so."
      > 对 2025 年的马里兰，影像不足以支持一个州级数字，我们如实说明。

14. We wrote down every test rule before we saw the answer. (can cut)
    > 每一条检验规则，我们都是在看到答案之前写下来的。（可删）

## Five likely questions

Two sentences each. Do not add to them; if the judge wants more, point to the limits page.

**1. Your model never saw US data. Why should it work here?**
It looks at how wet the crop is when it is cut, not at the calendar date, and that is the same in Europe and here. On the organisers' labelled fields it got 42 of 44 right with no US training.
> 它看的是收割时作物有多湿，而不是日历上的日期，这一点欧洲和这里是一样的。在主办方带标签的地块上，它没有用任何美国数据训练，44 个对了 42 个。

**2. Why such a small model? Did you try deep learning or a pretrained model?**
Yes, we tried bigger models and a pretrained satellite model. In a new country they lose more accuracy than the small one, and on the labelled fields the big model got 37 of 44 against 42.
> 试过，更大的模型和预训练的卫星模型都试了。换一个国家时，它们掉的精度比小模型多；在带标签的地块上，大模型是 44 对 37，小模型是 42。

**3. How accurate are your county and state acres?**
Not proven yet: our first county method failed its test in Kent County, where it said 21 percent silage and the Census says 5.5. The second method misses by at most 3.8 points on the four Maryland counties, but it was built on those same counties, so that is not a test. *(Add the Michigan sentence from slot A when it is logged.)*
> 还没有得到证实：第一种县级方法在 Kent 县没通过检验，它算出青贮占 21%，而普查是 5.5%。第二种方法在马里兰四个县上最多差 3.8 个百分点，但它就是在这四个县上做出来的，所以这不算检验。（密歇根结果出来后，加上 A 处那一句。）

**4. What do you do about clouds?**
If the last image before the cut and the first image after it are more than 20 days apart, the field is not judged. We count silage only where the harvest was seen, and next to every total we show the share of corn we could not see.
> 如果收割前最后一张影像和收割后第一张影像相隔超过 20 天，这块地就不判断。我们只在看到了收割的地方计青贮，并在每个总数旁边报告没看到的玉米占多少。

**5. Did you tune the model on the 44 labelled fields? Is 44 enough?**
No: the model was frozen on European data first, the set was scored twice by a plan written before the first run, and every look at it is in a log. 44 is a small test, one field is 2.3 points, so 42 and 43 would mean the same.
> 没有：模型先在欧洲数据上定型；这组地块按事先写好的计划打了两次分；每一次查看都记在日志里。44 个是个小测试，一块地就是 2.3 个百分点，所以 42 和 43 没有实质区别。

## Words to keep the same everywhere

| Say | Not | 中文 |
|---|---|---|
| clearly seen / partly seen / not judged | clear tier, wide tier, abstain | 看清了 / 部分看到 / 不判断 |
| the small model | logistic regression, eu2_logistic_v1 | 小模型（六个特征的逻辑回归） |
| the big model | LightGBM, GBM | 大模型（135 个特征的提升树） |
| first method / second method (for county totals) | local threshold, window weighting | 第一种 / 第二种县级加总方法 |
| corn we could not see | blind windows | 没看到的玉米 |
| 42 of 44 | 95 percent | 44 个里对 42 个 |
