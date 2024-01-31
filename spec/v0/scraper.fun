// import aiohttp
// import asyncio
// import requests
// import json
// from selenium.common.exceptions import ElementNotInteractableException, ElementClickInterceptedException, StaleElementReferenceException
// from selenium import webdriver
// from selenium.webdriver.firefox.service import Service as FirefoxService
// from webdriver_manager.firefox import GeckoDriverManager
// from selenium.webdriver.common.by import By
// import argparse
// from bs4 import BeautifulSoup
let http = use("http")
let fs = use("sys.fs")
let json = use("encoding.json")
let html = use("encoding.html")
let os = use("os")

// LINK_FILE = "./episode_slugs.txt"
// TRANSCRIPT_DIR = "./transcripts"
//
// PODCAST_NAME = 'lex-fridman-podcast-10'
// PODCAST_LINK = f'https://steno.ai/{PODCAST_NAME}'
// GET_PODCAST_PAGE_URL = f'{PODCAST_LINK}/fetch-podcast'
// print(GET_PODCAST_PAGE_URL)
//
// s = requests.Session()

let LINK_FILE = "./update_slugs.txt"
let TRANSCRIPT_DIR = "./transcripts"
let PODCAST_NAME = 'lex-fridman-podcast-10'
let PODCAST_LINK = f'https://steno.ai/{PODCAST_NAME}'
let GET_PODCAST_PAGE_URL = f'{PODCAST_LINK}/fetch-podcast'

// def get_podcast_page(page):
//     params = {'page': page}
//     res = s.get(GET_PODCAST_PAGE_URL, params=params)
//     return res.json()
// NOTE: type not required because it is inferred from params map
fun get_podcast_page(page) {
    let params = map[str : int] {
        page: page
    }
    // .json() defaults to map[str | int : any]
    let res = http.get(GET_PODCAST_PAGE_URL, params=params);
    let res_json: json.Object = json.parse(res.body);
}

// def print_status(*args):
//     print("\r\033[2K", end='')
//     print(*args, end='', flush=True)
fun print_clear(...args) {
    print("\r\033[2K", end='')
    print(...args, end='', flush=True)
}

// slugs = []
// if args.update_slugs:
//     if args.open_browser:
//         options = webdriver.FirefoxOptions()
//
//         # selenium 4
//         with webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options) as driver:
//             driver.get(PODCAST_LINK)
        
let selenium = use("selenium")

fun init_browser() {
    // NOTE: dynamic/deferred import
    let options = selenium.Options.FireFox {}
    let driver = selenium.FireFox()
    return driver
}

//             def at_last():
//                 # looks for this element on the page:
//                 # <h3>Sorry, there are no more episodes for this podcast.</h3>
//                 end_str = "Sorry, there are no more episodes for this podcast."
//                 elems = driver.find_elements(By.TAG_NAME, "h3")
//                 for elem in elems:
//                     if elem.text == end_str:
//                         return True
//                 return False
fun at_last(page: html.Element) {
    let end_str = "Sorry, there are no more episodes for this podcast."
    // html.find_all takes html.By enum (union)
    let h3s = page.find_all(html.By.Tag(html.Tag.h3))
    for h3 in h3s {
        if h3.text == end_str {
            return true
        }
    }
    return false
}
//             def close_popup():
//                 # closes the advertising popup that comes up occasionally
//                 driver.find_element(By.ID, "modal__close").click()
//                 return True
fun close_popup(driver) {
    driver.find(html.By.Id("modal__close"))?.click()
    return Ok()
}
//             loaded_more_count = 0
//             while True:
//                 try:
//                     driver.find_element(By.ID, "loadMoreButton").click()
//                     loaded_more_count += 1
//                     print('Loaded More: ', loaded_more_count, end='', flush=True)
//                     print("\r", end='')
//                 except ElementNotInteractableException:
//                     if at_last():
//                         break
//                     # wait for button to load
//                     continue
//                 except StaleElementReferenceException:
//                     if at_last():
//                         break
//                     # else:
//                     #     raise e
//                 except ElementClickInterceptedException:
//                     close_popup()
//                     continue
//             elems = driver.find_elements(By.CLASS_NAME, "article")
//             print(len(elems))
//             with open(LINK_FILE, 'r+') as f:
//                 existing_slugs = f.read().split('\n')
//                 print('existing slugs:', '\n'.join(slugs))
//                 for elem in elems:
//                     slug = elem.find_element(
//                         By.CLASS_NAME, "card").get_attribute("data-episode-slug").strip()
//                     if slug not in slugs:
//                         print('found new slug:', slug)
//                         slugs.append(slug)
//             slugs.sort()
//             print("PARSED SLUGS")
fun update_slugs_with_selenium(driver) {
    let loaded_more_count = 0
    loop {
        let loadMoreButton = driver.find(html.By.Id("loadMoreButton"));
        match loadMoreButton {
            Ok(driver.Clickable(btn)) => {
                match btn.click() {
                    Err(selenium.Error.StaleElementReference) => {
                        if at_last(driver.page) {
                            break
                        }
                        // return err
                        return loadMoreButton
                    }
                    Err(selenium.Error.ClickIntercepted) => {
                        close_popup(driver)
                        continue
                    }
                    _ => {}
                }
                loaded_more_count += 1
                print('Loaded More: ', loaded_more_count, end='', flush=True)
                print("\r", end='')
            },
            Ok(other) => {
                if at_last(driver.page) {
                    break
                }
                // wait for button to load
                continue
            }
        }
    }
    let link_file = os.open(LINK_FILE, os.WriteAppend)
    defer link_file.close()

    let elems = driver.find_all(html.By.Class("article"))
    print(elems.len())
    let existing_slugs = link_file.read()?.split('\n')
    print("existing slugs: \n\t", slugs.join("\n\t"))

    let new_slugs = []
    for elem in elems {
        let slug = elems.find(html.By.Class("card")).attrs["data-episode-slug"]?.strip()
        if slug not in existing_slugs {
            print(f"found new slug {slug}")
            new_slugs.push(slug)
        }
    }
    link_file.write(new_slugs.join("\n"))
}
//     else:
//         first_page = get_podcast_page(1)
//         last_page_num = first_page['episodes']['last_page']
//         print("Total Pages:", last_page_num)
//         for page_num in range(1, last_page_num):
//             page = get_podcast_page(page_num)['episodes']
//             podcasts = page['data']
//
//             page_slugs = [pod['slug'] for pod in podcasts]
//             print_status('Current Page:',
//                          page['current_page'], 'Podcast:', page_slugs[-1])
//             slugs += page_slugs
//
//     # common
//     with open(LINK_FILE, 'w') as f:
//         f.write('\n'.join(slugs))
fun update_slugs() {
    let slugs = []
    let first_page = get_podcast_page(1)
    let last_page_num = first_page["episodes"]?["last_page"]?
    for page_num in 1..last_page_num {
        let page = get_podcast_page(page_num)["episodes"]?
        let podcasts = page["data"]
        let page_slugs = podcasts.map(p => p["slug"])
        print_clear("Current Page:", page["current_page"], "Podcast:", page_slugs[-1])
        slugs += page_slugs
    }
}
// elif args.update_transcripts:
//     try:
//         with open(LINK_FILE, 'r') as f:
//             slugs = f.read().split('\n')
//     except FileNotFoundError:
//         import sys
//         print(f"{LINK_FILE} does not exist. Run this script again with the `-s` flag to create it", file=sys.stderr)
//         exit(1)
fun get_slugs() {
    let f = os.open(LINK_FILE, os.Read)
    if Err(os.FileNotFound) = f {
        print(f"{LINK_FILE} does not exist. Run this script again with the `-s` flag to create it", file=sys.stderr)
        let slugs = f.read().lines()
        sys.exit(1)
    }
}
//
// def to_link(slug):
//     return PODCAST_LINK + '/' + slug

fun to_link(slug) {
    return PODCAST_LINK + '/' + slug
}

// def to_path(slug):
//     return TRANSCRIPT_DIR + '/' + slug + '.json'

fun to_path(slug) {
    return TRANSCRIPT_DIR + '/' + slug + ".json"
}

// async def get_slug_page_content(slug, session):
//     url = to_link(slug)
//     async with session.get(url) as response:
//         return await response.content.read()
fun get_slug_page_content(slug) {
    let url = to_link(slug)
    let response = http.get(url)
    return html.parse(response.content.read())
}
//
// def parse_transcript(page_content):
//     soup = BeautifulSoup(page_content, features='html.parser')
//     passages = soup.find_all(class_="transcript")
//     transcript = {}
//     for passage in passages:
//         try:
//             info = {}
//             timestamp = passage['id']
//             info['timestamp'] = timestamp
//             info['text'] = passage.p.mark.text
//             transcript[timestamp] = info
//         except:
//             print("Error reading transcript")
//             continue
//     return transcript

fun parse_transcript(page) {
    let passages = page.find_all(html.By.Class("transcript"))
    let transcript = map[str : str]
    for passage in passages {
        let timestamp = passage.attrs["id"]
        let text = passage.find(html.By.Tag("p")).text
        transcript[timestamp] = text
        
    }
}
// async def update_slug_transcript(slug, session):
//     content = await get_slug_page_content(slug, session)
//     transcript = await asyncio.to_thread(parse_transcript,content)
//     json.dump(transcript, open(to_path(slug), 'w'), indent=4)
//     print("Parsed:", slug)
//     
fun update_slug_transcript(slug) {
    let content = get_slug_page_content(slug)
    let transcript = parse_transcript(content)
    let json_str = json.encode(transcript)
    let f = os.open(to_path(slug), os.Write)
    f.write(json_str)
    f.close()
    print("Parsed:", slug)
}
// async def update_transcripts():
//     tasks = []
//     async with aiohttp.ClientSession() as session:
//         for slug in slugs:
//             tasks.append(asyncio.create_task(update_slug_transcript(slug, session)))
//         await asyncio.gather(*tasks)
//             
So tree.extra_data[lhs] is params_start, tree.extra_data[lhs+1] is params_end, and so on until tree.extra_data[lhs+5] is callconv_expr.

let sync = use("sync")
fun update_transcripts() {
    let tasks = []
    let wg = sync.WaitGroup
    for slug in slugs {
        wg.add(1)
        // NOTE: godo
        godo(update_slug_transcript(slug)).then(wg.done())
    }
    wg.wait()
}

// parser = argparse.ArgumentParser()
// parser.add_argument('-s', '--slugs', action='store_true', dest='update_slugs')
// parser.add_argument('-t', '--transcripts',
//                     action='store_true', dest='update_transcripts')
// parser.add_argument('-o', '--open-browser',
//                     dest='open_browser', action='store_true')
// args = parser.parse_args()
let flags = use("flags")

let parser = flags.parser.new()

parser.add_arg("update_slugs", flags.BoolArg {
    short: "-s",
    long: "--slugs",
})

parser.add_arg("update_transcripts", flag.BoolArg {
    short: "-t",
    long: "--transcripts",
})

let args = parser.parse_args()

// if args.update_transcripts:
//     asyncio.run(update_transcripts())
fun main() {
    let slugs = match (args.update_slugs, args.open_browser) {
        (true, true) => update_slugs_with_selenium(),
        (true, false) => update_slugs(),
        (false, false) => get_slugs()
    }
    if args.update_transcripts {
        update_transcripts()
    }
}
