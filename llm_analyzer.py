    import asyncio
    import json
    import os
    import re
    import sys
    import shutil
    from pathlib import Path
    from typing import List, Dict, Optional, Tuple, Any, Set
    
    import httpx
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    PATENTS_DIR = Path("patents")
    CLOUD_DIR = Path("patents_cloud")
    OUTPUT_DIR = Path("datasets")
    OUTPUT_DIR.mkdir(exist_ok=True)
    CLOUD_DIR.mkdir(exist_ok=True)
    
    OUTPUT_JSON = OUTPUT_DIR / "all_alloys.json"
    PROGRESS_JSON = OUTPUT_DIR / "progress.json"
    
    _LLM_SEMAPHORE = asyncio.Semaphore(3)
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    
    _llm_client: Optional[AsyncOpenAI] = None
    
    ALLOY_CATEGORIES = {
        "nickel_alloys": "Ni",
        "aluminum": "Al",
        "titanium_alloys": "Ti",
        "copper_alloys": "Cu",
        "steel_alloys": "Fe",
        "magnesium_alloys": "Mg",
        "cobalt_alloys": "Co",
    }
    
    
    def clean_filename(filename: str) -> str:
        """Очищает имя файла от недопустимых символов для безопасного использования в файловой системе Windows"""
        if not filename:
            return "unknown"
    
        invalid_chars = r'[<>:"/\\|?*\n\r\t]'
        cleaned = re.sub(invalid_chars, "_", filename)
    
        cleaned = cleaned.strip()
    
        if not cleaned or cleaned == "." or cleaned == "..":
            cleaned = "unknown"
    
        if len(cleaned) > 200:
            name, ext = os.path.splitext(cleaned)
            cleaned = name[:200] + ext
    
        return cleaned
    
    
    class YandexDiskClient:
        """Клиент для работы с API Яндекс.Диска: скачивание файлов и навигация по папкам"""
    
        API_BASE = "https://cloud-api.yandex.net/v1/disk"
    
        def __init__(self, token: Optional[str] = None):
            self.token = token or os.getenv("YANDEX_DISK_TOKEN")
            if not self.token:
                raise ValueError("YANDEX_DISK_TOKEN not found in .env")
            self.headers = {
                "Authorization": f"OAuth {self.token}",
                "Accept": "application/json",
            }
            self.api_client = httpx.AsyncClient(headers=self.headers, timeout=60.0)
            import requests
    
            self.sync_session = requests.Session()
            self.sync_session.headers.update(self.headers)
    
        async def close(self):
            """Закрывает HTTP-сессии клиента"""
            await self.api_client.aclose()
            self.sync_session.close()
    
        async def list_folder(self, disk_path: str) -> Tuple[List[Dict], List[Dict]]:
            """Получает список папок и файлов по указанному пути на Яндекс.Диске"""
            url = f"{self.API_BASE}/resources"
            params = {"path": disk_path, "limit": 1000, "sort": "name"}
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await self.api_client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("_embedded", {}).get("items", [])
                    return [i for i in items if i.get("type") == "dir"], [
                        i for i in items if i.get("type") == "file"
                    ]
                except Exception as e:
                    print(f"   Warning reading Yandex.Disk folder (attempt {attempt + 1}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
            return [], []
    
        async def get_download_link(self, disk_path: str) -> Optional[str]:
            """Получает временную ссылку для скачивания файла с Яндекс.Диска"""
            url = f"{self.API_BASE}/resources/download"
            params = {"path": disk_path}
            try:
                resp = await self.api_client.get(url, params=params)
                resp.raise_for_status()
                return resp.json().get("href")
            except Exception as e:
                print(f"   Failed to get download link for {disk_path}: {e}")
                return None
    
        def _download_sync(self, file_url: str, local_path: Path) -> bool:
            """Синхронное скачивание файла по URL (выполняется в отдельном потоке)"""
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with self.sync_session.get(file_url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return True
            except Exception as e:
                print(f"   Error downloading {local_path.name}: {str(e)[:80]}")
                return False
    
        async def download_file(self, file_url: str, local_path: Path) -> bool:
            """Асинхронно скачивает файл, делегируя синхронную загрузку в поток"""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._download_sync, file_url, local_path
            )
    
        async def download_patent_folder(
            self, disk_path: str, local_folder: Path
        ) -> List[Path]:
            """Рекурсивно скачивает папку с патентом с Яндекс.Диска в локальную директорию"""
            downloaded: List[Path] = []
            url = f"{self.API_BASE}/resources"
            params = {"path": disk_path, "limit": 1000}
            try:
                resp = await self.api_client.get(url, params=params)
                resp.raise_for_status()
                items = resp.json().get("_embedded", {}).get("items", [])
            except Exception as e:
                print(f"   Failed to read {disk_path}: {e}")
                return downloaded
    
            for item in items:
                item_name = item["name"]
                item_path = item["path"]
    
                item_name_clean = clean_filename(item_name)
                local_file_path = local_folder / item_name_clean
    
                if item["type"] == "dir":
                    try:
                        if not local_file_path.exists():
                            local_file_path.mkdir(parents=True, exist_ok=True)
                        sub_downloaded = await self.download_patent_folder(
                            item_path, local_file_path
                        )
                        downloaded.extend(sub_downloaded)
                    except Exception as e:
                        print(f"      Error creating folder {item_name_clean}: {e}")
                        continue
    
                elif item["type"] == "file":
                    try:
                        download_url = await self.get_download_link(item_path)
                        if download_url:
                            if await self.download_file(download_url, local_file_path):
                                downloaded.append(local_file_path)
                                print(
                                    f"      Downloaded {item_name_clean[:50]}{'...' if len(item_name_clean) > 50 else ''}"
                                )
                    except Exception as e:
                        print(f"      Error downloading file {item_name_clean}: {e}")
                        continue
    
            return downloaded
    
    
    def get_client() -> AsyncOpenAI:
        """Инициализирует и возвращает асинхронный клиент DeepSeek API (синглтон)"""
        global _llm_client
        if _llm_client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip(
                "/"
            )
            if not api_key:
                print("\nDEEPSEEK_API_KEY not found in .env")
                sys.exit(1)
            _llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
            print("DeepSeek API initialized")
        return _llm_client
    
    
    def read_patent_files(patent_path: Path) -> Tuple[str, List[str]]:
        """Читает все текстовые файлы в папке патента и объединяет их содержимое"""
        all_text, files_read = "", []
        for file_path in patent_path.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in [
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".xml",
                ".html",
            ]:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="cp1251") as f:
                        content = f.read()
                except Exception:
                    continue
            except Exception:
                continue
            rel = file_path.relative_to(patent_path)
            all_text += f"\n\n{'=' * 60}\nFILE: {rel}\n{'=' * 60}\n\n{content}"
            files_read.append(str(rel))
        return all_text, files_read
    
    
    def parse_value(val: Any) -> Optional[float]:
        """Парсит одиночное числовое значение из строки или числа, удаляя единицы измерения"""
        if val is None:
            return None
    
        if isinstance(val, (int, float)):
            if val > 0:
                return float(val)
            return None
    
        val_str = str(val).strip().replace(",", ".")
    
        val_str = re.sub(
            r"\s*(mm|cm|m|km|MPa|kgf|C|F|K|deg)$",
            "",
            val_str,
            flags=re.IGNORECASE,
        )
    
        try:
            f = float(val_str)
            if f > 0:
                return round(f, 6)
        except ValueError:
            pass
    
        return None
    
    
    def parse_range(val: Any) -> Optional[Dict[str, float]]:
        """Парсит строку или число в диапазон значений с поддержкой форматов 'min-max', '<X', '≤X', '>X'"""
        if val is None:
            return None
    
        val_str = str(val).strip().replace(",", ".")
    
        val_str = re.sub(
            r"\s*(mm|cm|m|km|MPa|kgf|C|F|K|deg)$",
            "",
            val_str,
            flags=re.IGNORECASE,
        )
    
        try:
            f = float(val_str)
            if f <= 0:
                return None
            return {"min": round(f, 6), "max": round(f, 6)}
        except ValueError:
            pass
    
        range_match = re.match(r"^\s*([\d.]+)\s*[-–÷~]\s*([\d.]+)\s*$", val_str)
        if range_match:
            a, b = float(range_match.group(1)), float(range_match.group(2))
            if a <= 0 and b <= 0:
                return None
            return {"min": round(min(a, b), 6), "max": round(max(a, b), 6)}
    
        lt_match = re.match(r"^<\s*([\d.]+)$", val_str)
        if lt_match:
            upper = float(lt_match.group(1))
            return {"min": 0.0, "max": round(upper, 6)}
    
        le_match = re.match(r"^≤\s*([\d.]+)$", val_str)
        if le_match:
            upper = float(le_match.group(1))
            return {"min": 0.0, "max": round(upper, 6)}
        gt_match = re.match(r"^>\s*([\d.]+)$", val_str)
        if gt_match:
            lower = float(gt_match.group(1))
            return {"min": round(lower, 6), "max": 100.0}
    
        return None
    
    
    def get_analysis_prompt(
        patent_text: str, patent_folder: str, alloy_category: str
    ) -> str:
        """Формирует промпт для LLM, instructing модель извлечь структурированные данные о сплаве из патента"""
        text = patent_text[:30000]
    
        return f"""You are a patent and materials science expert. Extract alloy data from the patent.
    
    PATENT FOLDER: {patent_folder}
    ALLOY CATEGORY: {alloy_category}
    
    =============================================================================
    KEY REQUIREMENT: YOU MUST FIND MECHANICAL PROPERTIES
    =============================================================================
    
    PATENT TEXT:
    {text}
    
    =============================================================================
    RETURN ONLY JSON OBJECT:
    =============================================================================
    
    {{
      "patent_number": "BY0000006232C1",
      "alloy_type": "nickel_alloy",
      "composition": {{
        "ni": {{"min": 60.0, "max": 75.0}},
        "cr": {{"min": 15.0, "max": 25.0}}
      }},
      "mechanical_properties": {{
        "sigma_u": {{"min": 750, "max": 850}},
        "sigma_y": {{"min": 600, "max": 700}},
        "elongation": {{"min": 12, "max": 18}},
        "hardness": {{"min": 280, "max": 320, "unit": "HB"}}
      }},
      "heat_treatment": {{
        "type": "solution_aging",
        "solution_temp": 1100,
        "aging_temp": 800,
        "cooling": "air"
      }},
      "application": "turbine_blades",
      "operating_temperature": {{"min": 800, "max": 950}}
    }}
    
    =============================================================================
    SEARCHING FOR MECHANICAL PROPERTIES (MOST IMPORTANT):
    =============================================================================
    
    1. WHERE TO SEARCH:
       - Tables named "Mechanical properties"
       - Tables named "Alloy properties"
       - Sections "Industrial applicability"
       - Sections "Examples"
       - CSV files: Table_1.csv, Table_2.csv, etc.
    
    2. HOW TO FIND sigma_u (ultimate tensile strength):
       - Keywords: "Rm", "UTS", "sb", "tensile strength"
       - Units: "MPa", "kgf/mm²" (×9.8 = MPa)
       - Examples in text:
         * "Ultimate tensile strength is 750-850 MPa"
         * "Rm = 800 MPa"
         * "Tensile strength: 750-850 MPa"
    
    3. HOW TO FIND hardness:
       - Keywords: "HB", "HRC", "HV", "hardness"
       - Examples:
         * "Brinell hardness 280-320 HB"
         * "HB 300"
    
    4. IF MULTIPLE VALUES EXIST:
       - For different temperatures: take range from min to max
       - For different conditions: take all values
    
    5. IF NO MECHANICAL PROPERTIES:
       - Set mechanical_properties: null (BUT FIRST CHECK ALL TABLES)
    
    =============================================================================
    OTHER RULES:
    =============================================================================
    
    1. COMPOSITION: total 100% (±1%)
    2. HEAT TREATMENT: temperature in C, time in hours
    3. APPLICATION: alloy application
    4. OPERATING TEMPERATURE: in C
    
    =============================================================================
    RETURN ONLY JSON, NO EXPLANATIONS"""
    
    
    async def call_llm_with_retry(prompt: str) -> Optional[Dict]:
        """Асинхронно вызывает LLM API с повторными попытками при сбоях (retry-механизм)"""
        client = get_client()
        for attempt in range(MAX_RETRIES):
            async with _LLM_SEMAPHORE:
                try:
                    response = await client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a patent expert. You MUST search for mechanical properties (sigma_u, hardness) in tables. Respond only with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=4000,
                    )
                    result_text = response.choices[0].message.content.strip()
                    result_text = re.sub(r"^```json\s*", "", result_text)
                    result_text = re.sub(r"^```\s*", "", result_text)
                    result_text = re.sub(r"\s*```$", "", result_text)
    
                    if '"mechanical_properties"' in result_text:
                        if '"sigma_u"' in result_text:
                            print(f"   Mechanical properties found")
                        else:
                            print(f"   mechanical_properties present but no sigma_u")
                    else:
                        print(f"   mechanical_properties missing from LLM response")
    
                    parsed = json.loads(result_text)
    
                    if parsed == {}:
                        return None
                    return parsed
                except json.JSONDecodeError as e:
                    print(
                        f"   Attempt {attempt + 1}/{MAX_RETRIES}: Invalid JSON: {e}"
                    )
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                except Exception as e:
                    print(
                        f"   Attempt {attempt + 1}/{MAX_RETRIES}: {type(e).__name__}: {e}"
                    )
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
        return None
    
    
    def normalize_composition(comp: Dict) -> Dict:
        """Нормализует химический состав: приводит ключи к нижнему регистру и корректирует сумму до 100%"""
        if not comp:
            return {}
    
        clean_comp = {}
        for element, value in comp.items():
            if isinstance(value, dict) and "min" in value and "max" in value:
                try:
                    mn, mx = float(value["min"]), float(value["max"])
                    if mx > 0:
                        clean_comp[element.lower()] = {"min": max(0.0, mn), "max": mx}
                except:
                    continue
            else:
                parsed = parse_range(value)
                if parsed:
                    clean_comp[element.lower()] = parsed
    
        if clean_comp:
            total_min = sum(v["min"] for v in clean_comp.values())
            total_max = sum(v["max"] for v in clean_comp.values())
    
            if total_max < 95 or total_max > 105:
                rest_elem = None
                for elem in clean_comp:
                    if elem in ["bal", "rest", "balance"]:
                        rest_elem = elem
                        break
    
                if rest_elem:
                    others_min = sum(
                        v["min"] for k, v in clean_comp.items() if k != rest_elem
                    )
                    others_max = sum(
                        v["max"] for k, v in clean_comp.items() if k != rest_elem
                    )
                    clean_comp[rest_elem] = {
                        "min": max(0.0, 100.0 - others_max),
                        "max": max(0.0, 100.0 - others_min),
                    }
    
        return clean_comp
    
    
    def parse_mechanical_properties(props: Dict) -> Optional[Dict]:
        """Парсит механические свойства из ответа LLM и приводит к единому формату"""
        if not props or not isinstance(props, dict):
            return None
    
        clean_props = {}
    
        if "sigma_u" in props and props["sigma_u"] is not None:
            sigma_u_val = props["sigma_u"]
            if isinstance(sigma_u_val, dict):
                if "min" in sigma_u_val and "max" in sigma_u_val:
                    try:
                        mn = float(sigma_u_val["min"])
                        mx = float(sigma_u_val["max"])
                        if mx > 0:
                            clean_props["sigma_u"] = {
                                "min": round(mn, 1),
                                "max": round(mx, 1),
                            }
                    except:
                        pass
                elif "min" in sigma_u_val:
                    try:
                        val = float(sigma_u_val["min"])
                        if val > 0:
                            clean_props["sigma_u"] = {
                                "min": round(val, 1),
                                "max": round(val, 1),
                            }
                    except:
                        pass
            else:
                parsed = parse_range(sigma_u_val)
                if parsed:
                    clean_props["sigma_u"] = parsed
    
        if "sigma_y" in props and props["sigma_y"] is not None:
            sigma_y_val = props["sigma_y"]
            if isinstance(sigma_y_val, dict):
                if "min" in sigma_y_val and "max" in sigma_y_val:
                    try:
                        mn = float(sigma_y_val["min"])
                        mx = float(sigma_y_val["max"])
                        if mx > 0:
                            clean_props["sigma_y"] = {
                                "min": round(mn, 1),
                                "max": round(mx, 1),
                            }
                    except:
                        pass
                elif "min" in sigma_y_val:
                    try:
                        val = float(sigma_y_val["min"])
                        if val > 0:
                            clean_props["sigma_y"] = {
                                "min": round(val, 1),
                                "max": round(val, 1),
                            }
                    except:
                        pass
            else:
                parsed = parse_range(sigma_y_val)
                if parsed:
                    clean_props["sigma_y"] = parsed
    
        if "elongation" in props and props["elongation"] is not None:
            elong_val = props["elongation"]
            if isinstance(elong_val, dict):
                if "min" in elong_val and "max" in elong_val:
                    try:
                        mn = float(elong_val["min"])
                        mx = float(elong_val["max"])
                        if mx > 0:
                            clean_props["elongation"] = {
                                "min": round(mn, 1),
                                "max": round(mx, 1),
                            }
                    except:
                        pass
                elif "min" in elong_val:
                    try:
                        val = float(elong_val["min"])
                        if val > 0:
                            clean_props["elongation"] = {
                                "min": round(val, 1),
                                "max": round(val, 1),
                            }
                    except:
                        pass
            else:
                parsed = parse_range(elong_val)
                if parsed:
                    clean_props["elongation"] = parsed
    
        if "hardness" in props and props["hardness"] is not None:
            hardness_val = props["hardness"]
            if isinstance(hardness_val, dict):
                if "min" in hardness_val and "max" in hardness_val:
                    try:
                        mn = float(hardness_val["min"])
                        mx = float(hardness_val["max"])
                        unit = hardness_val.get("unit", "HB")
                        unit = re.sub(r"\d+", "", unit)
                        if mx > 0:
                            clean_props["hardness"] = {
                                "min": round(mn, 1),
                                "max": round(mx, 1),
                                "unit": unit if unit in ["HB", "HRC", "HV"] else "HB",
                            }
                    except:
                        pass
                elif "min" in hardness_val:
                    try:
                        val = float(hardness_val["min"])
                        unit = hardness_val.get("unit", "HB")
                        unit = re.sub(r"\d+", "", unit)
                        if val > 0:
                            clean_props["hardness"] = {
                                "min": round(val, 1),
                                "max": round(val, 1),
                                "unit": unit if unit in ["HB", "HRC", "HV"] else "HB",
                            }
                    except:
                        pass
            else:
                parsed = parse_range(hardness_val)
                if parsed:
                    clean_props["hardness"] = {**parsed, "unit": "HB"}
    
        return clean_props if clean_props else None
    
    
    def parse_heat_treatment(ht: Dict) -> Optional[Dict]:
        """Парсит параметры термической обработки из ответа LLM (тип, температуры, время, охлаждение)"""
        if not ht or not isinstance(ht, dict):
            return None
    
        clean_ht = {}
    
        if "type" in ht and ht["type"]:
            clean_ht["type"] = str(ht["type"]).lower()
    
        if "solution_temp" in ht:
            try:
                clean_ht["solution_temp"] = float(ht["solution_temp"])
            except:
                pass
    
        if "solution_time" in ht:
            stime = ht["solution_time"]
            if isinstance(stime, dict) and "min" in stime and "max" in stime:
                try:
                    clean_ht["solution_time"] = {
                        "min": float(stime["min"]),
                        "max": float(stime["max"]),
                        "unit": stime.get("unit", "hours"),
                    }
                except:
                    pass
            else:
                parsed = parse_range(stime)
                if parsed:
                    clean_ht["solution_time"] = {**parsed, "unit": "hours"}
    
        if "aging_temp" in ht:
            try:
                clean_ht["aging_temp"] = float(ht["aging_temp"])
            except:
                pass
    
        if "aging_time" in ht:
            atime = ht["aging_time"]
            if isinstance(atime, dict) and "min" in atime and "max" in atime:
                try:
                    clean_ht["aging_time"] = {
                        "min": float(atime["min"]),
                        "max": float(atime["max"]),
                        "unit": atime.get("unit", "hours"),
                    }
                except:
                    pass
            else:
                parsed = parse_range(atime)
                if parsed:
                    clean_ht["aging_time"] = {**parsed, "unit": "hours"}
    
        if "cooling" in ht and ht["cooling"]:
            cooling = str(ht["cooling"]).lower()
            if cooling in ["water", "oil", "air", "furnace"]:
                clean_ht["cooling"] = cooling
    
        return clean_ht if clean_ht else None
    
    
    def create_empty_record(patent_number: str, alloy_type: str) -> Dict:
        """Создаёт пустую запись сплава с заданной структурой и полями со значением None"""
        return {
            "patent_number": patent_number,
            "alloy_type": alloy_type,
            "composition": {},
            "mechanical_properties": None,
            "heat_treatment": None,
            "application": None,
            "operating_temperature": None,
        }
    
    
    def clean_record(record: Dict, existing_ids: Optional[Set] = None) -> Optional[Dict]:
        """Очищает и нормализует запись от LLM, обрабатывает дубликаты patent_number и приводит к строгой структуре"""
        if not record or record == {}:
            return None
    
        record_id = record.get("patent_number") or ""
        if not record_id:
            return None
    
        if existing_ids is None:
            existing_ids = set()
        if record_id in existing_ids:
            counter = 1
            new_id = f"{record_id}_{counter}"
            while new_id in existing_ids:
                counter += 1
                new_id = f"{record_id}_{counter}"
            record_id = new_id
        existing_ids.add(record_id)
    
        alloy_type = record.get("alloy_type") or ""
    
        result = create_empty_record(record_id, alloy_type)
    
        comp = record.get("composition") or {}
        clean_comp = normalize_composition(comp)
        if clean_comp:
            result["composition"] = clean_comp
    
        mechanical_properties = record.get("mechanical_properties")
        clean_mechanical = parse_mechanical_properties(mechanical_properties)
        if clean_mechanical:
            result["mechanical_properties"] = clean_mechanical
    
        heat_treatment = record.get("heat_treatment")
        clean_heat_treatment = parse_heat_treatment(heat_treatment)
        if clean_heat_treatment:
            result["heat_treatment"] = clean_heat_treatment
    
        application = record.get("application")
        if application and isinstance(application, str):
            result["application"] = application.strip()
    
        operating_temperature = record.get("operating_temperature")
        clean_operating_temp = (
            parse_range(operating_temperature) if operating_temperature else None
        )
        if clean_operating_temp:
            result["operating_temperature"] = clean_operating_temp
    
        if not result["composition"]:
            return None
    
        return result
    
    
    def print_extraction_result(clean: Dict):
        """Выводит в консоль информацию об успешно извлечённой записи сплава"""
        print(f"   Added: {clean.get('patent_number')}")
    
        comp = clean.get("composition", {})
        if comp:
            comp_preview = {
                k: f"{v['min']:.1f}-{v['max']:.1f}" for k, v in list(comp.items())[:4]
            }
            print(f"   Composition: {comp_preview}")
            if len(comp) > 4:
                print(f"      ... and {len(comp) - 4} more elements")
        else:
            print(f"   Composition: No data")
    
        mech = clean.get("mechanical_properties")
        if mech:
            props = []
            if mech.get("sigma_u"):
                props.append(
                    f"sigma_u={mech['sigma_u']['min']:.0f}-{mech['sigma_u']['max']:.0f}MPa"
                )
            if mech.get("sigma_y"):
                props.append(
                    f"sigma_y={mech['sigma_y']['min']:.0f}-{mech['sigma_y']['max']:.0f}MPa"
                )
            if mech.get("elongation"):
                props.append(
                    f"delta={mech['elongation']['min']:.0f}-{mech['elongation']['max']:.0f}%"
                )
            if mech.get("hardness"):
                h = mech["hardness"]
                props.append(f"{h['unit']}={h['min']:.0f}-{h['max']:.0f}")
            if props:
                print(f"   Mechanical properties: {', '.join(props)}")
            else:
                print(f"   Mechanical properties: {list(mech.keys())}")
        else:
            print(f"   Mechanical properties: No data")
    
        ht = clean.get("heat_treatment")
        if ht:
            ht_str = []
            if ht.get("type"):
                ht_str.append(ht["type"])
            if ht.get("solution_temp"):
                ht_str.append(f"solution {ht['solution_temp']:.0f}C")
            if ht.get("aging_temp"):
                ht_str.append(f"aging {ht['aging_temp']:.0f}C")
            if ht_str:
                print(f"   Heat treatment: {', '.join(ht_str)}")
        else:
            print(f"   Heat treatment: No data")
    
        app = clean.get("application")
        if app:
            app_short = app[:80] + "..." if len(app) > 80 else app
            print(f"   Application: {app_short}")
        else:
            print(f"   Application: No data")
    
        op_temp = clean.get("operating_temperature")
        if op_temp:
            print(f"   Operating temperature: {op_temp['min']:.0f}-{op_temp['max']:.0f}C")
        else:
            print(f"   Operating temperature: No data")
    
    
    def load_json_dataset() -> List[Dict]:
        """Загружает существующий JSON-датасет из файла all_alloys.json"""
        if OUTPUT_JSON.exists():
            try:
                with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [d for d in data if d and d.get("composition")]
            except Exception as e:
                print(f"Warning reading {OUTPUT_JSON}: {e}")
        return []
    
    
    def save_json_dataset(data: List[Dict]):
        """Сохраняет датасет в JSON-файл all_alloys.json"""
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    
    def load_progress() -> dict:
        """Загружает файл progress.json с информацией об уже обработанных патентах и их категориях"""
        if PROGRESS_JSON.exists():
            try:
                with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"processed_folders": [], "folder_categories": {}}
    
    
    def save_progress(processed_folders: list, folder_categories: dict):
        """Сохраняет прогресс обработки в файл progress.json для возможности возобновления после сбоя"""
        with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed_folders": processed_folders,
                    "folder_categories": folder_categories,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    
    async def process_single_patent(
        patent_path: Path,
        progress_id: str,
        folder_category: str,
        dataset: List[Dict],
        processed_folders: list,
        folder_categories: dict,
    ) -> int:
        """Обрабатывает один патент: читает файлы, вызывает LLM, очищает результат и добавляет в датасет"""
        try:
            print(f"\n{progress_id} [{folder_category}]")
    
            patent_text, files_read = read_patent_files(patent_path)
            if not patent_text or len(patent_text) < 100:
                print(f"   No text data")
                if progress_id not in processed_folders:
                    processed_folders.append(progress_id)
                    folder_categories[progress_id] = folder_category
                    save_progress(processed_folders, folder_categories)
                return 0
    
            print(f"   Files: {len(files_read)} | Characters: {len(patent_text)}")
    
            alloy_category = folder_category
            for key in ALLOY_CATEGORIES.keys():
                if key in progress_id.lower():
                    alloy_category = key
                    break
    
            prompt = get_analysis_prompt(patent_text, progress_id, alloy_category)
            result = await call_llm_with_retry(prompt)
    
            if result is None:
                print(f"   Not an alloy or empty result")
                if progress_id not in processed_folders:
                    processed_folders.append(progress_id)
                    folder_categories[progress_id] = folder_category
                    save_progress(processed_folders, folder_categories)
                return 0
    
            current_ids = {
                r.get("patent_number") for r in dataset if r.get("patent_number")
            }
            clean = clean_record(result, existing_ids=current_ids)
    
            if clean:
                if not clean.get("alloy_type") and folder_category:
                    clean["alloy_type"] = folder_category
    
                dataset.append(clean)
                save_json_dataset(dataset)
                print_extraction_result(clean)
            else:
                print(f"   No valid data (possibly not an alloy)")
    
            if progress_id not in processed_folders:
                processed_folders.append(progress_id)
                folder_categories[progress_id] = folder_category
                save_progress(processed_folders, folder_categories)
    
            return 1 if clean else 0
    
        except Exception as e:
            print(f"   CRITICAL ERROR processing {progress_id}: {e}")
            if progress_id not in processed_folders:
                processed_folders.append(progress_id)
                folder_categories[progress_id] = folder_category
                save_progress(processed_folders, folder_categories)
            return 0
    
    
    async def process_yandex_patents(
        dataset: List[Dict], processed_folders: list, folder_categories: dict
    ) -> int:
        """Загружает патенты с Яндекс.Диска по категориям и обрабатывает их через process_single_patent"""
        yandex_token = os.getenv("YANDEX_DISK_TOKEN")
        if not yandex_token:
            print("\nYANDEX_DISK_TOKEN not found - skipping cloud patents")
            return 0
    
        yandex_path = os.getenv("YANDEX_DISK_PATH", "disk:/patents_alloys")
        yandex_path = yandex_path.replace("https://", "").replace(
            "disk.yandex.ru/client/disk/", "disk:/"
        )
    
        print(f"\n{'=' * 70}")
        print(f"YANDEX.DISK: {yandex_path}")
        print(f"{'=' * 70}")
    
        yd = YandexDiskClient(yandex_token)
        try:
            alloy_folders, files = await yd.list_folder(yandex_path)
        except Exception as e:
            print(f"Failed to connect to Yandex.Disk: {e}")
            await yd.close()
            return 0
    
        print(f"   Alloy categories: {len(alloy_folders)}")
    
        total_added = 0
    
        for idx, alloy_folder in enumerate(alloy_folders, 1):
            folder_name = alloy_folder["name"]
            disk_path = alloy_folder["path"]
    
            folder_name_clean = clean_filename(folder_name)
    
            category = "unknown"
            for cat in ALLOY_CATEGORIES.keys():
                if cat in folder_name_clean.lower():
                    category = cat
                    break
    
            print(f"\n{'=' * 60}")
            print(
                f"[{idx}/{len(alloy_folders)}] Category: {folder_name_clean} -> {category}"
            )
            print(f"{'=' * 60}")
    
            try:
                patent_folders, patent_files = await yd.list_folder(disk_path)
            except Exception as e:
                print(f"   Error reading folder {folder_name_clean}: {e}")
                continue
    
            if not patent_folders:
                print(f"   No patents in category {folder_name_clean}")
                continue
    
            print(f"   Patents in category: {len(patent_folders)}")
    
            for patent_idx, patent_folder in enumerate(patent_folders, 1):
                patent_name = patent_folder["name"]
                patent_name_clean = clean_filename(patent_name)
                patent_disk_path = patent_folder["path"]
                progress_id = f"yd_{folder_name_clean}_{patent_name_clean}"
    
                if progress_id in processed_folders:
                    print(
                        f"\n[{patent_idx}/{len(patent_folders)}] {patent_name_clean} - already processed"
                    )
                    continue
    
                print(
                    f"\n[{patent_idx}/{len(patent_folders)}] Patent: {patent_name_clean}"
                )
    
                local_patent_folder = CLOUD_DIR / folder_name_clean / patent_name_clean
    
                try:
                    if local_patent_folder.exists():
                        shutil.rmtree(local_patent_folder)
                    local_patent_folder.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    print(f"   Error creating folder {local_patent_folder}: {e}")
                    print(f"   Skipping patent {patent_name_clean}")
                    if progress_id not in processed_folders:
                        processed_folders.append(progress_id)
                        folder_categories[progress_id] = category
                        save_progress(processed_folders, folder_categories)
                    continue
    
                try:
                    downloaded = await yd.download_patent_folder(
                        patent_disk_path, local_patent_folder
                    )
                except Exception as e:
                    print(f"   Error downloading patent {patent_name_clean}: {e}")
                    if progress_id not in processed_folders:
                        processed_folders.append(progress_id)
                        folder_categories[progress_id] = category
                        save_progress(processed_folders, folder_categories)
                    continue
    
                if not downloaded:
                    print(f"   No files to download")
                    if progress_id not in processed_folders:
                        processed_folders.append(progress_id)
                        folder_categories[progress_id] = category
                        save_progress(processed_folders, folder_categories)
                    continue
    
                print(f"   Downloaded: {len(downloaded)} files")
    
                try:
                    added = await process_single_patent(
                        local_patent_folder,
                        progress_id,
                        category,
                        dataset,
                        processed_folders,
                        folder_categories,
                    )
                    total_added += added
                except Exception as e:
                    print(f"   Error processing patent {patent_name_clean}: {e}")
                    if progress_id not in processed_folders:
                        processed_folders.append(progress_id)
                        folder_categories[progress_id] = category
                        save_progress(processed_folders, folder_categories)
    
                await asyncio.sleep(1)
    
        await yd.close()
        return total_added
    
    
    async def process_local_patents(
        dataset: List[Dict], processed_folders: list, folder_categories: dict
    ) -> int:
        """Обрабатывает локальные патенты из папки patents, сгруппированные по категориям"""
        if not PATENTS_DIR.exists():
            print(f"\nLocal folder {PATENTS_DIR} not found")
            return 0
    
        category_folders = [p for p in PATENTS_DIR.iterdir() if p.is_dir()]
        category_folders.sort()
    
        print(f"\n{'=' * 70}")
        print(f"LOCAL PATENTS: {len(category_folders)} categories")
        print(f"{'=' * 70}")
    
        total_added = 0
    
        for cat_idx, cat_path in enumerate(category_folders, 1):
            category_name = cat_path.name
            alloy_category = "unknown"
            for cat in ALLOY_CATEGORIES.keys():
                if cat in category_name.lower():
                    alloy_category = cat
                    break
    
            print(f"\n{'=' * 60}")
            print(
                f"[{cat_idx}/{len(category_folders)}] Category: {category_name} -> {alloy_category}"
            )
            print(f"{'=' * 60}")
    
            try:
                patent_folders = [p for p in cat_path.iterdir() if p.is_dir()]
                patent_folders.sort()
            except Exception as e:
                print(f"   Error reading category {category_name}: {e}")
                continue
    
            if not patent_folders:
                print(f"   No patents in category {category_name}")
                continue
    
            print(f"   Patents in category: {len(patent_folders)}")
    
            for patent_idx, patent_path in enumerate(patent_folders, 1):
                patent_name = clean_filename(patent_path.name)
                progress_id = f"local_{category_name}_{patent_name}"
    
                if progress_id in processed_folders:
                    print(
                        f"\n[{patent_idx}/{len(patent_folders)}] {patent_name} - already processed"
                    )
                    continue
    
                print(f"\n[{patent_idx}/{len(patent_folders)}] Patent: {patent_name}")
    
                try:
                    added = await process_single_patent(
                        patent_path,
                        progress_id,
                        alloy_category,
                        dataset,
                        processed_folders,
                        folder_categories,
                    )
                    total_added += added
                except Exception as e:
                    print(f"   Error processing patent {patent_name}: {e}")
                    if progress_id not in processed_folders:
                        processed_folders.append(progress_id)
                        folder_categories[progress_id] = alloy_category
                        save_progress(processed_folders, folder_categories)
    
                await asyncio.sleep(0.5)
    
        return total_added
    
    
    async def main():
        """Главная функция модуля: загружает прогресс, обрабатывает локальные патенты и патенты с Яндекс.Диска, сохраняет результаты"""
        print("\n" + "=" * 70)
        print("LLM ANALYZER JSON - Alloy Data Collection")
        print(
            "   Structure: patent_number | alloy_type | composition | mechanical_properties | heat_treatment | application | operating_temperature"
        )
        print("   null = no data")
        print("=" * 70)
    
        dataset = load_json_dataset()
        progress_data = load_progress()
        processed_folders = progress_data.get("processed_folders", [])
        folder_categories = progress_data.get("folder_categories", {})
    
        print(f"\nCurrent dataset: {len(dataset)} records")
        print(f"Already processed patents: {len(processed_folders)}")
    
        local_added = await process_local_patents(
            dataset, processed_folders, folder_categories
        )
        cloud_added = await process_yandex_patents(
            dataset, processed_folders, folder_categories
        )
    
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print(f"   Local records added: {local_added}")
        print(f"   Cloud records added: {cloud_added}")
        print(f"   Total in dataset: {len(dataset)}")
        print(f"   File: {OUTPUT_JSON}")
    
        if dataset:
            alloy_types = {}
            for record in dataset:
                at = record.get("alloy_type", "unknown")
                alloy_types[at] = alloy_types.get(at, 0) + 1
    
            print(f"\nAlloy type statistics:")
            for at, count in sorted(alloy_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   {at}: {count}")
    
            total = len(dataset)
            has_mech = sum(1 for r in dataset if r.get("mechanical_properties"))
            has_ht = sum(1 for r in dataset if r.get("heat_treatment"))
            has_app = sum(1 for r in dataset if r.get("application"))
            has_temp = sum(1 for r in dataset if r.get("operating_temperature"))
    
            print(f"\nData availability in dataset ({total} records):")
            print(f"   Mechanical properties: {has_mech} ({has_mech * 100 // total}%)")
            print(f"   Heat treatment: {has_ht} ({has_ht * 100 // total}%)")
            print(f"   Application: {has_app} ({has_app * 100 // total}%)")
            print(f"   Operating temperature: {has_temp} ({has_temp * 100 // total}%)")
    
            sigmas = []
            for r in dataset:
                mech = r.get("mechanical_properties")
                if mech and mech.get("sigma_u"):
                    sigma_u = mech["sigma_u"]
                    sigmas.append((sigma_u["min"] + sigma_u["max"]) / 2)
    
            if sigmas:
                print(f"\nSigma_u statistics (range average):")
                print(
                    f"   Min: {min(sigmas):.1f} | Max: {max(sigmas):.1f} | Mean: {sum(sigmas) / len(sigmas):.1f}"
                )
    
    
    if __name__ == "__main__":
        asyncio.run(main())