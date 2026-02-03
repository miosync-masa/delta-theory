#!/usr/bin/env python3
"""
FatigueData-AM2022 → Upstash Redis アップロードスクリプト

Usage:
    1. FatigueData-AM2022.json をダウンロード
       curl -L -o FatigueData-AM2022.json "https://ndownloader.figshare.com/files/41023481"
    
    2. 環境変数を設定
       export UPSTASH_URL="https://your-redis.upstash.io"
       export UPSTASH_TOKEN="your-token"
    
    3. 実行
       python upload_to_redis.py FatigueData-AM2022.json

Author: 環 & ご主人さま (飯泉真道)
License: MIT
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any

try:
    from upstash_redis import Redis
except ImportError:
    print("❌ upstash-redis not installed!")
    print("   pip install upstash-redis")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

# da/dN データは大きいので分割格納
DADN_CHUNK_SIZE = 50000  # 1チャンクあたりの最大データ点数


# =============================================================================
# Parser
# =============================================================================

def parse_fatigue_data(json_path: str) -> dict[str, Any]:
    """
    FatigueData-AM2022.json をパースして構造化データを返す
    """
    print(f"📂 Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    root = data['fatiguedata_am2022']
    articles = root['articles']
    print(f"   {len(articles)} articles found")
    
    # 材料ごとにデータを集約
    materials_data = defaultdict(lambda: {
        'sn': [],
        'en': [],
        'dadn': [],
        'sigma_y_range': [float('inf'), 0],
        'sigma_uts_range': [float('inf'), 0],
    })
    
    for art in articles:
        doi = art['metadata'].get('doi', '')
        datasets = art['scidata']['datasets']
        
        for ds in datasets:
            mat_name = ds.get('materials', {}).get('mat_name', 'unknown')
            fat = ds.get('fatigue', {})
            ftype = fat.get('fdata_type', 'unknown')
            
            test = ds.get('testing', {})
            stat = ds.get('static_mech', {})
            
            # 試験条件
            R = test.get('fat_r', [None])[0] if test.get('fat_r') else None
            sigma_y = stat.get('yield_strength', [None])[0] if stat.get('yield_strength') else None
            sigma_uts = stat.get('tensile_strength', [None])[0] if stat.get('tensile_strength') else None
            E_mod = stat.get('modulus', [None])[0] if stat.get('modulus') else None
            
            # σ_y, σ_uts 範囲更新
            if sigma_y:
                materials_data[mat_name]['sigma_y_range'][0] = min(
                    materials_data[mat_name]['sigma_y_range'][0], sigma_y)
                materials_data[mat_name]['sigma_y_range'][1] = max(
                    materials_data[mat_name]['sigma_y_range'][1], sigma_y)
            
            if sigma_uts:
                materials_data[mat_name]['sigma_uts_range'][0] = min(
                    materials_data[mat_name]['sigma_uts_range'][0], sigma_uts)
                materials_data[mat_name]['sigma_uts_range'][1] = max(
                    materials_data[mat_name]['sigma_uts_range'][1], sigma_uts)
            
            fat_data = fat.get('fat_data', [])
            
            # S-N データ (fdata_type == 'sn')
            if ftype == 'sn' and fat_data:
                for pt in fat_data:
                    N, S, runout = pt[0], pt[1], pt[2] if len(pt) > 2 else 0
                    materials_data[mat_name]['sn'].append({
                        'N': N,
                        'S': S,
                        'runout': int(runout),
                        'R': R,
                        'sigma_y': sigma_y,
                        'sigma_uts': sigma_uts,
                        'doi': doi
                    })
            
            # ε-N データ (fdata_type == 'en')
            elif ftype == 'en' and fat_data:
                for pt in fat_data:
                    N, eps_a = pt[0], pt[1]
                    runout = pt[2] if len(pt) > 2 else 0
                    materials_data[mat_name]['en'].append({
                        'N': N,
                        'eps_a': eps_a,
                        'runout': int(runout),
                        'R': R,
                        'sigma_y': sigma_y,
                        'E': E_mod,
                        'doi': doi
                    })
            
            # da/dN データ (fdata_type == 'dadn')
            elif ftype == 'dadn' and fat_data:
                for pt in fat_data:
                    dadn, dK = pt[0], pt[1]
                    materials_data[mat_name]['dadn'].append({
                        'dadn': dadn,
                        'dK': dK,
                        'R': R,
                        'doi': doi
                    })
    
    return dict(materials_data)


# =============================================================================
# Redis Upload
# =============================================================================

def upload_to_redis(materials_data: dict[str, Any], redis: Redis) -> dict[str, int]:
    """
    パース済みデータをUpstash Redisに格納
    """
    print("\n📤 Uploading to Upstash Redis...")
    
    stored = {'sn': 0, 'en': 0, 'dadn': 0}
    
    # 材料サマリー
    materials_summary = {}
    
    for mat_name, mat_data in materials_data.items():
        sn_count = len(mat_data['sn'])
        en_count = len(mat_data['en'])
        dadn_count = len(mat_data['dadn'])
        
        if sn_count + en_count + dadn_count == 0:
            continue
        
        # サマリー作成
        materials_summary[mat_name] = {
            'sn_count': sn_count,
            'en_count': en_count,
            'dadn_count': dadn_count,
            'sigma_y_min': mat_data['sigma_y_range'][0] if mat_data['sigma_y_range'][0] != float('inf') else None,
            'sigma_y_max': mat_data['sigma_y_range'][1] if mat_data['sigma_y_range'][1] != 0 else None,
            'sigma_uts_min': mat_data['sigma_uts_range'][0] if mat_data['sigma_uts_range'][0] != float('inf') else None,
            'sigma_uts_max': mat_data['sigma_uts_range'][1] if mat_data['sigma_uts_range'][1] != 0 else None,
        }
        
        # S-N データ格納
        if sn_count > 0:
            redis.set(f'fatigue:sn:{mat_name}', json.dumps(mat_data['sn']))
            stored['sn'] += sn_count
            print(f"   ✓ fatigue:sn:{mat_name} ({sn_count} points)")
        
        # ε-N データ格納
        if en_count > 0:
            redis.set(f'fatigue:en:{mat_name}', json.dumps(mat_data['en']))
            stored['en'] += en_count
            print(f"   ✓ fatigue:en:{mat_name} ({en_count} points)")
        
        # da/dN データ格納（大きい場合は分割）
        if dadn_count > 0:
            if dadn_count <= DADN_CHUNK_SIZE:
                redis.set(f'fatigue:dadn:{mat_name}', json.dumps(mat_data['dadn']))
                print(f"   ✓ fatigue:dadn:{mat_name} ({dadn_count} points)")
            else:
                # 分割格納
                chunks = [mat_data['dadn'][i:i+DADN_CHUNK_SIZE] 
                         for i in range(0, dadn_count, DADN_CHUNK_SIZE)]
                for idx, chunk in enumerate(chunks):
                    redis.set(f'fatigue:dadn:{mat_name}:{idx}', json.dumps(chunk))
                    print(f"   ✓ fatigue:dadn:{mat_name}:{idx} ({len(chunk)} points)")
                # チャンク数を記録
                materials_summary[mat_name]['dadn_chunks'] = len(chunks)
            stored['dadn'] += dadn_count
    
    # 材料サマリー格納
    redis.set('fatigue:materials', json.dumps(materials_summary))
    print(f"   ✓ fatigue:materials ({len(materials_summary)} materials)")
    
    # メタ情報格納
    from datetime import datetime
    meta = {
        'source': 'FatigueData-AM2022',
        'doi': '10.1038/s41597-023-02150-x',
        'figshare_doi': '10.6084/m9.figshare.22337629',
        'license': 'CC BY 4.0',
        'total_materials': len(materials_summary),
        'total_sn_points': stored['sn'],
        'total_en_points': stored['en'],
        'total_dadn_points': stored['dadn'],
        'updated': datetime.now().strftime('%Y-%m-%d'),
        'description': 'Fatigue database of additively manufactured alloys'
    }
    redis.set('fatigue:meta', json.dumps(meta))
    print(f"   ✓ fatigue:meta")
    
    return stored


# =============================================================================
# Main
# =============================================================================

def main():
    # 引数チェック
    if len(sys.argv) < 2:
        print("Usage: python upload_to_redis.py <FatigueData-AM2022.json>")
        print("")
        print("環境変数:")
        print("  UPSTASH_URL   - Upstash Redis URL")
        print("  UPSTASH_TOKEN - Upstash Redis token")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        sys.exit(1)
    
    # 環境変数チェック
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        print("❌ Environment variables not set!")
        print("   export UPSTASH_URL='https://your-redis.upstash.io'")
        print("   export UPSTASH_TOKEN='your-token'")
        sys.exit(1)
    
    # Redis接続
    print("🔌 Connecting to Upstash Redis...")
    redis = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)
    
    # 接続テスト
    try:
        redis.set('fatigue:test', 'connection_ok')
        redis.delete('fatigue:test')
        print("   ✓ Connected!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # パース
    materials_data = parse_fatigue_data(json_path)
    
    # アップロード
    stored = upload_to_redis(materials_data, redis)
    
    # サマリー
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"   Materials:    {len(materials_data)}")
    print(f"   S-N points:   {stored['sn']:,}")
    print(f"   ε-N points:   {stored['en']:,}")
    print(f"   da/dN points: {stored['dadn']:,}")
    print(f"   TOTAL:        {sum(stored.values()):,}")


if __name__ == '__main__':
    main()
